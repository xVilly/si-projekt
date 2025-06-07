import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from numba import jit
import time
import os
import sys
from PIL import Image

# Make sure these files are in the same directory or your PYTHONPATH
from setup_mnist_model import MNIST
from setup_cifar10_model import CIFAR10

"""
L2 Black Box Attack - Refactored for Robustness
"""

# Numba-jitted functions for performance-critical coordinate updates.
# These operate on NumPy arrays and MUST be run on the CPU.

@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size, beta1, beta2, proj):
    """
    Performs the ADAM update for a batch of coordinates.
    """
    # Estimate gradients using finite differences
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002

    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt

    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt

    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))

    m = real_modifier.ravel() # Use ravel() for a contiguous 1D view
    old_val = m[indice]
    old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)

    # Project to valid range if not using tanh
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])

    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, real_modifier, up, down, step_size, proj):
    """
    Performs the Newton update for a batch of coordinates.
    """
    # In this implementation, the original losses[0] is not used correctly for hessian.
    # We will stick to the original logic, but note it might be suboptimal.
    # A better approach would be to pass the loss of the unmodified input.
    cur_loss = losses[0]
    
    # Estimate gradients and Hessians using finite differences
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (0.0001 * 0.0001)

    # Sanitize Hessian values
    hess[hess < 0] = 1.0  # Avoid negative curvature
    hess.clip(min=0.1, out=hess) # Avoid division by zero or very small numbers

    m = real_modifier.ravel() # Use ravel() for a contiguous 1D view
    old_val = m[indice]
    old_val -= step_size * grad / hess

    # Project to valid range if not using tanh
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])

    m[indice] = old_val

def loss_run(input_tensor, target_label, model, modifier, use_tanh, use_log, targeted, confidence, const):
    """
    Calculates the attack loss for a batch of perturbations.
    
    Shapes:
        - input_tensor: (N, C, H, W)
        - target_label: (N, num_classes)
        - modifier: (N, C, H, W)
    Returns:
        - Tensors for losses, scores, and perturbed images.
    """
    if use_tanh:
        # Tanh maps the perturbation to [-1, 1], then scale to [-0.5, 0.5]
        perturbed_input = torch.tanh(input_tensor + modifier) / 2
        # The L2 distance is measured against the original image in the [-0.5, 0.5] space
        original_input_space = torch.tanh(input_tensor) / 2
    else:
        perturbed_input = input_tensor + modifier
        original_input_space = input_tensor

    output = model(perturbed_input)
    
    if use_log:
        output = F.softmax(output, dim=-1)

    # L2 distance loss
    l2_dist = torch.sum(torch.square(perturbed_input - original_input_space), dim=(1, 2, 3))

    # Classification loss
    real = torch.sum(target_label * output, dim=-1)
    other, _ = torch.max((1 - target_label) * output - (target_label * 10000), dim=-1)

    if use_log:
        real = torch.log(real + 1e-30)
        other = torch.log(other + 1e-30)

    # The confidence parameter should be a tensor on the same device
    confidence_tensor = torch.tensor(confidence, dtype=torch.float32, device=input_tensor.device)

    if targeted:
        # For targeted attacks, we want to maximize the target class score (real)
        # and minimize the score of all other classes (other).
        # Loss is high if (other - real) is not less than -confidence.
        classification_loss = torch.max(other - real, -confidence_tensor)
    else:
        # For untargeted attacks, we want to minimize the original class score (real)
        # and maximize any other class score (other).
        # Loss is high if (real - other) is not less than -confidence.
        classification_loss = torch.max(real - other, -confidence_tensor)

    classification_loss = const * classification_loss
    total_loss = l2_dist + classification_loss
    
    # Return detached numpy arrays for the optimization functions
    return (
        total_loss.detach().cpu().numpy(),
        l2_dist.detach().cpu().numpy(),
        classification_loss.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
        perturbed_input.detach().cpu().numpy()
    )


def l2_attack(input_img, target_label, model, device, targeted, use_log, use_tanh, solver,
              reset_adam_after_found=True, abort_early=True, batch_size=128, max_iter=1000,
              const=0.01, confidence=0.0, early_stop_iters=100, binary_search_steps=5,
              step_size=0.01, adam_beta1=0.9, adam_beta2=0.999):
    """
    Performs the L2 attack on a single image.
    
    Shapes:
        - input_img: (1, C, H, W) - Expects a single image with a batch dimension.
        - target_label: (1, num_classes)
    Returns:
        - A single adversarial image of shape (C, H, W)
    """
    # --- Shape and Device Validation ---
    assert input_img.shape[0] == 1, "l2_attack is designed to run on a single image."
    input_tensor = torch.tensor(input_img, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_label, dtype=torch.float32, device=device)
    
    # --- Initialization ---
    img_shape = input_tensor.shape[1:] # (C, H, W)
    var_len = input_tensor.numel() # Total number of pixels
    
    real_modifier = torch.zeros((1,) + img_shape, dtype=torch.float32, device=device)
    
    # ADAM optimizer state (must be numpy for numba)
    mt = np.zeros(var_len, dtype=np.float32)
    vt = np.zeros(var_len, dtype=np.float32)
    adam_epoch = np.ones(var_len, dtype=np.int32)
    
    # Buffers for gradient/hessian estimation
    grad = np.zeros(batch_size, dtype=np.float32)
    hess = np.zeros(batch_size, dtype=np.float32)

    # Binary search for constant c
    lower_bound = 0.0
    upper_bound = 1e10
    
    # ==============================================================================
    # THE FIX IS HERE: Initialize with shape (C, H, W) to ensure consistency
    # ==============================================================================
    out_best_attack = input_img[0].copy() 
    # ==============================================================================

    out_best_const = const
    out_bestl2 = float('inf')
    out_bestscore = -1

    if use_tanh:
        input_tensor = torch.atanh(input_tensor * 1.99999)

    modifier_up = np.full(var_len, 0.5, dtype=np.float32) - input_tensor.cpu().numpy().ravel()
    modifier_down = np.full(var_len, -0.5, dtype=np.float32) - input_tensor.cpu().numpy().ravel()
    
    def compare(score, target_idx):
        if not isinstance(score, (float, int, np.int64)):
            pred_idx = np.argmax(score)
        else:
            pred_idx = int(score)
        if targeted:
            return pred_idx == target_idx
        else:
            return pred_idx != target_idx

    target_class_idx = np.argmax(target_label)

    for step in range(binary_search_steps):
        print(f"  Binary search step {step+1}/{binary_search_steps}, current const={const:.6f}")
        
        bestl2_at_step = float('inf')
        bestscore_at_step = -1
        last_loss2 = 1.0
        
        mt.fill(0)
        vt.fill(0)
        adam_epoch.fill(1)
        real_modifier.fill_(0)
        
        stage = 0
        
        for iter_idx in range(max_iter):
            indices = np.random.choice(var_len, batch_size, replace=False)

            modifier_numpy = real_modifier.cpu().numpy()
            eval_modifiers_np = np.repeat(modifier_numpy, batch_size * 2 + 1, axis=0)

            for i in range(batch_size):
                eval_modifiers_np[i * 2 + 1].ravel()[indices[i]] += 0.0001
                eval_modifiers_np[i * 2 + 2].ravel()[indices[i]] -= 0.0001
            
            eval_modifiers_torch = torch.from_numpy(eval_modifiers_np).to(device)

            losses, l2s, losses2, scores, pert_images = loss_run(
                input_tensor, target_tensor.repeat(len(eval_modifiers_torch), 1), model,
                eval_modifiers_torch, use_tanh, use_log, targeted, confidence, const
            )

            if solver == "adam":
                coordinate_ADAM(losses, indices, grad, hess, batch_size, mt, vt, modifier_numpy, adam_epoch,
                                modifier_up, modifier_down, step_size, adam_beta1, adam_beta2, proj=not use_tanh)
            elif solver == "newton":
                newton_losses = np.concatenate(([losses[0]], losses[1:]))
                coordinate_Newton(newton_losses, indices, grad, hess, batch_size, modifier_numpy,
                                  modifier_up, modifier_down, step_size, proj=not use_tanh)
            else:
                raise ValueError(f"Unknown solver: {solver}")

            real_modifier = torch.from_numpy(modifier_numpy).to(device)

            if compare(scores[0], target_class_idx):
                if l2s[0] < bestl2_at_step:
                    bestl2_at_step = l2s[0]
                    bestscore_at_step = np.argmax(scores[0])
                
                if l2s[0] < out_bestl2:
                    print(f"    [PROGRESS] New best attack found! iter={iter_idx+1}, L2={l2s[0]:.4f}, loss={losses[0]:.4f}")
                    out_bestl2 = l2s[0]
                    out_bestscore = np.argmax(scores[0])
                    # pert_images[0] has shape (C, H, W), which is now consistent with initialization
                    out_best_attack = pert_images[0]
                    out_best_const = const

            if (iter_idx + 1) % 100 == 0:
                print(f"    iter={iter_idx+1}, loss={losses[0]:.5f}, l2={l2s[0]:.5f}, c_loss={losses2[0]:.5f}")
                sys.stdout.flush()
        
        if bestscore_at_step != -1:
            print(f"  SUCCESS: Attack found with const={const:.6f}. Lowering const.")
            upper_bound = min(upper_bound, const)
            const = (lower_bound + upper_bound) / 2
        else:
            print(f"  FAILURE: Attack not found with const={const:.6f}. Increasing const.")
            lower_bound = max(lower_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            else:
                const *= 10
    
    return out_best_attack, out_bestscore

def generate_data(test_loader, targeted, samples, start, num_classes):
    """Generates a batch of inputs and targets for the attack."""
    inputs, targets = [], []
    count = 0
    for i, (data, label) in enumerate(test_loader):
        if i < start:
            continue
        if count >= samples:
            break
        
        if targeted:
            # For each image, create a target for all other classes
            for j in range(num_classes):
                if j == label.item():
                    continue
                inputs.append(data[0].numpy())
                targets.append(np.eye(num_classes)[j])
        else:
            # For untargeted, the target is the original class label
            inputs.append(data[0].numpy())
            targets.append(np.eye(num_classes)[label.item()])
        count += 1
        
    return np.array(inputs), np.array(targets)


def attack_main(inputs, targets, model, device, hparams):
    """
    Main attack loop that iterates over all input samples.
    """
    r = []
    print(f'Starting attack on {len(inputs)} samples...')
    for i in range(len(inputs)):
        print(f"\n--- Attacking sample {i+1}/{len(inputs)} ---")
        
        # The attack function expects a batch dimension, so we add one.
        # Shape: (1, C, H, W)
        input_single = np.expand_dims(inputs[i], 0)
        target_single = np.expand_dims(targets[i], 0)
        
        attack_img, score = l2_attack(
            input_single, target_single, model, device,
            targeted=hparams['targeted'],
            use_log=hparams['use_log'],
            use_tanh=hparams['use_tanh'],
            solver=hparams['solver'],
            batch_size=hparams['batch_size'],
            max_iter=hparams['max_iter'],
            binary_search_steps=hparams['binary_search_steps']
        )
        r.append(attack_img)
    
    # Stack results. The shape will be (N, C, H, W) because l2_attack returns a single image.
    return np.stack(r)


if __name__ == '__main__':
    # =================================================================
    #                 CONFIGURATION PARAMETERS
    # =================================================================
    HPARAMS = {
        "dataset": "cifar10",  # "cifar10" or "mnist"
        "targeted": True,
        "solver": "adam",     # "adam" or "newton"
        "use_log": True,
        "use_tanh": True,
        "samples_to_attack": 1, # Number of source images to attack
        "start_from_sample": 0, # Offset in the test dataset
        "batch_size": 128,      # Batch size for gradient estimation, NOT image batching
        "max_iter": 100,       # Max iterations per binary search step
        "binary_search_steps": 3 # Number of steps to find the optimal const
    }
    
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Device Setup ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Model and Data Loading ---
    if HPARAMS['dataset'] == 'mnist':
        # MNIST images are normalized to [-0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        model = MNIST().to(device)
        model.load_state_dict(torch.load('./models/mnist_model.pt', map_location=device))
        num_classes = 10
        classes = [str(i) for i in range(10)]
    elif HPARAMS['dataset'] == 'cifar10':
        # CIFAR-10 images are normalized to [-0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))])
        test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        model = CIFAR10().to(device)
        model.load_state_dict(torch.load('./models/cifar10_model.pt', map_location=device))
        num_classes = 10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError("Invalid dataset choice.")
        
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    model.eval()

    # --- Generate Attack Data ---
    inputs, targets = generate_data(
        test_loader, HPARAMS['targeted'], HPARAMS['samples_to_attack'], HPARAMS['start_from_sample'], num_classes
    )
    print(f"Generated {len(inputs)} attack instances.")

    # --- Run Attack ---
    timestart = time.time()
    adv = attack_main(inputs, targets, model, device, HPARAMS)
    timeend = time.time()
    print(f"\nAttack finished. Took {(timeend - timestart) / 60.0:.2f} mins for {len(inputs)} samples.")

    # --- Evaluation ---
    # The attack returns shape (N, C, H, W). This is the correct shape for the model.
    # The original code had a bug where adv had shape (N, 1, C, H, W). My refactoring fixes this.
    assert adv.shape == inputs.shape, f"Shape mismatch! Adversarial: {adv.shape}, Original: {inputs.shape}"

    inputs_tensor = torch.from_numpy(inputs).to(device)
    adv_tensor = torch.from_numpy(adv).to(device)

    with torch.no_grad():
        if HPARAMS['use_log']:
            # Softmax is already applied in the loss function if use_log=True
            original_logits = F.softmax(model(inputs_tensor), dim=-1)
            adv_logits = F.softmax(model(adv_tensor), dim=-1)
        else:
            original_logits = model(inputs_tensor)
            adv_logits = model(adv_tensor)
            
    original_class = torch.argmax(original_logits, 1).cpu().numpy()
    adv_class = torch.argmax(adv_logits, 1).cpu().numpy()
    target_class = np.argmax(targets, 1)
    
    success_rate = np.mean(adv_class == target_class) if HPARAMS['targeted'] else np.mean(adv_class != original_class)
    
    print("\n--- Results ---")
    print(f"Original Labels:      {original_class}")
    print(f"Adversarial Labels:   {adv_class}")
    if HPARAMS['targeted']:
        print(f"Target Labels:        {target_class}")
    print(f"Attack Success Rate:  {success_rate * 100.0:.2f}%")
    print(f"Total L2 distortion:  {np.sum((adv - inputs)**2)**.5:.4f}")

    # --- Visualization ---
    plt.figure(figsize=(10, 10))
    num_to_show = min(len(adv), 25) # Show up to 25 examples
    for i in range(num_to_show):
        plt.subplot(5, 5, i + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        
        orig_label = classes[original_class[i]]
        adv_label = classes[adv_class[i]]
        plt.title(f"{orig_label} -> {adv_label}", fontsize=10)
        
        # De-normalize image from [-0.5, 0.5] to [0, 1] for plotting
        img = adv[i] + 0.5
        if HPARAMS['dataset'] == 'mnist':
            plt.imshow(img.squeeze(), cmap="gray")
        else: # cifar10
            plt.imshow(np.transpose(img, (1, 2, 0)))

    plt.tight_layout()
    filename = f"{HPARAMS['solver']}_{'targeted' if HPARAMS['targeted'] else 'untargeted'}_{HPARAMS['dataset']}.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
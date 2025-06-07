"""
L2 Black Box Attack - Modernized for PyTorch 2.x and CUDA
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from numba import jit
import time
import os
import sys

# Make sure these model definition files are present in the same directory
from setup_mnist_model import MNIST
from setup_cifar10_model import CIFAR10

# Numba-jitted functions remain unchanged as they operate on CPU NumPy arrays.
@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, adam_epoch, up, down, step_size, beta1, beta2, proj):
    """Performs the ADAM update for a batch of coordinates on the CPU."""
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.ravel()
    old_val = m[indice]
    old_val -= step_size * corr * mt / (np.sqrt(vt) + 1e-8)
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1

@jit(nopython=True)
def coordinate_Newton(losses, indice, grad, hess, batch_size, real_modifier, up, down, step_size, proj):
    """Performs the Newton update for a batch of coordinates on the CPU."""
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (0.0001 * 0.0001)
    hess[hess < 0] = 1.0
    hess.clip(min=0.1, out=hess)
    m = real_modifier.ravel()
    old_val = m[indice]
    old_val -= step_size * grad / hess
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val

def loss_run(input_tensor, target_label, model, modifier, use_tanh, use_log, targeted, confidence, const):
    """
    Calculates the attack loss. All tensor operations are performed on the specified device.
    
    Shapes:
        - input_tensor: (N, C, H, W) on device
        - target_label: (N, num_classes) on device
        - modifier: (N, C, H, W) on device
    """
    if use_tanh:
        perturbed_input = torch.tanh(input_tensor + modifier) / 2
        original_input_space = torch.tanh(input_tensor) / 2
    else:
        perturbed_input = input_tensor + modifier
        original_input_space = input_tensor

    output = model(perturbed_input)
    
    if use_log:
        output = F.softmax(output, dim=-1)

    l2_dist = torch.sum(torch.square(perturbed_input - original_input_space), dim=(1, 2, 3))

    real = torch.sum(target_label * output, dim=-1)
    other = torch.max((1 - target_label) * output - (target_label * 10000), dim=-1)[0]

    if use_log:
        real = torch.log(real + 1e-30)
        other = torch.log(other + 1e-30)

    confidence_tensor = torch.tensor(confidence, dtype=torch.float32, device=input_tensor.device)

    if targeted:
        classification_loss = torch.max(other - real, -confidence_tensor)
    else:
        classification_loss = torch.max(real - other, -confidence_tensor)

    classification_loss = const * classification_loss
    total_loss = l2_dist + classification_loss
    
    # Return detached numpy arrays (moved to CPU) for the optimization functions
    return (
        total_loss.detach().cpu().numpy(),
        l2_dist.detach().cpu().numpy(),
        classification_loss.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
        perturbed_input.detach().cpu().numpy()
    )

def l2_attack(input_img, target_label, model, device, hparams):
    """
    Performs the L2 attack on a single image, managing GPU/CPU data transfer.
    
    Shapes:
        - input_img: (1, C, H, W) - NumPy array from data loader.
        - target_label: (1, num_classes) - NumPy array.
    Returns:
        - A single adversarial image of shape (C, H, W) as a NumPy array.
    """
    # Unpack hyperparameters
    targeted = hparams['targeted']
    use_log = hparams['use_log']
    use_tanh = hparams['use_tanh']
    solver = hparams['solver']
    batch_size = hparams['batch_size']
    max_iter = hparams['max_iter']
    binary_search_steps = hparams['binary_search_steps']
    const = hparams['const']
    confidence = hparams['confidence']
    step_size = hparams['step_size']
    
    # --- Shape and Device Validation ---
    assert input_img.shape[0] == 1, "l2_attack is designed for a single image."
    # Move inputs to the target device (GPU)
    input_tensor = torch.tensor(input_img, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_label, dtype=torch.float32, device=device)
    
    # --- Initialization ---
    img_shape = input_tensor.shape[1:] # (C, H, W)
    var_len = input_tensor.numel()
    
    # Modifier is a tensor that lives on the GPU
    real_modifier = torch.zeros((1,) + img_shape, dtype=torch.float32, device=device)
    
    # ADAM/Newton state arrays must be on the CPU for Numba
    mt = np.zeros(var_len, dtype=np.float32)
    vt = np.zeros(var_len, dtype=np.float32)
    adam_epoch = np.ones(var_len, dtype=np.int32)
    grad = np.zeros(batch_size, dtype=np.float32)
    hess = np.zeros(batch_size, dtype=np.float32)

    # Binary search state
    lower_bound = 0.0
    upper_bound = 1e10
    
    # CORRECTED: Initialize with a (C, H, W) NumPy array to prevent shape errors
    out_best_attack = input_img[0].copy()
    out_best_const = const
    out_bestl2 = float('inf')
    out_bestscore = -1

    if use_tanh:
        input_tensor = torch.atanh(input_tensor * 1.99999)

    # Bounds for the modifier must also be CPU-based NumPy arrays
    modifier_up = 0.5 - input_tensor.cpu().numpy().ravel()
    modifier_down = -0.5 - input_tensor.cpu().numpy().ravel()
    
    def compare(score, target_idx):
        pred_idx = np.argmax(score) if not isinstance(score, (int, np.int64)) else int(score)
        return pred_idx == target_idx if targeted else pred_idx != target_idx

    target_class_idx = np.argmax(target_label)

    for step in range(binary_search_steps):
        print(f"  Binary search step {step+1}/{binary_search_steps}, current const={const:.6f}")
        
        bestl2_at_step, bestscore_at_step = float('inf'), -1
        mt.fill(0); vt.fill(0); adam_epoch.fill(1); real_modifier.fill_(0)
        
        for iter_idx in range(max_iter):
            indices = np.random.choice(var_len, batch_size, replace=False)

            # --- Data Transfer: GPU -> CPU ---
            # Create perturbations in NumPy on the CPU
            modifier_numpy = real_modifier.cpu().numpy()
            eval_modifiers_np = np.repeat(modifier_numpy, batch_size * 2 + 1, axis=0)
            for i in range(batch_size):
                eval_modifiers_np[i * 2 + 1].ravel()[indices[i]] += 0.0001
                eval_modifiers_np[i * 2 + 2].ravel()[indices[i]] -= 0.0001
            
            # --- Data Transfer: CPU -> GPU ---
            # Move perturbations to the GPU for model inference
            eval_modifiers_torch = torch.from_numpy(eval_modifiers_np).to(device)

            losses, l2s, losses2, scores, pert_images = loss_run(
                input_tensor, target_tensor.repeat(len(eval_modifiers_torch), 1), model,
                eval_modifiers_torch, use_tanh, use_log, targeted, confidence, const
            )

            # --- Optimization on CPU ---
            # losses are already on CPU from loss_run
            if solver == "adam":
                coordinate_ADAM(losses, indices, grad, hess, batch_size, mt, vt, modifier_numpy, adam_epoch,
                                modifier_up, modifier_down, step_size, 0.9, 0.999, proj=not use_tanh)
            else: # newton
                coordinate_Newton(losses, indices, grad, hess, batch_size, modifier_numpy,
                                  modifier_up, modifier_down, step_size, proj=not use_tanh)

            # --- Data Transfer: CPU -> GPU ---
            # Update the GPU modifier tensor with the new CPU values
            real_modifier = torch.from_numpy(modifier_numpy).to(device)

            # --- Check for Success and Update Best ---
            if compare(scores[0], target_class_idx):
                if l2s[0] < bestl2_at_step:
                    bestl2_at_step, bestscore_at_step = l2s[0], np.argmax(scores[0])
                if l2s[0] < out_bestl2:
                    print(f"    [PROGRESS] New best attack! iter={iter_idx+1}, L2={l2s[0]:.4f}, loss={losses[0]:.4f}")
                    out_bestl2, out_bestscore = l2s[0], np.argmax(scores[0])
                    # pert_images[0] is (C, H, W) NumPy array, consistent with initialization
                    out_best_attack = pert_images[0]
                    out_best_const = const

            if (iter_idx + 1) % 100 == 0:
                print(f"    iter={iter_idx+1}, loss={losses[0]:.5f}, l2={l2s[0]:.5f}, c_loss={losses2[0]:.5f}")

        # --- Update Constant for Binary Search ---
        if bestscore_at_step != -1:
            print(f"  SUCCESS with const={const:.6f}. Lowering const.")
            upper_bound = min(upper_bound, const)
            const = (lower_bound + upper_bound) / 2
        else:
            print(f"  FAILURE with const={const:.6f}. Increasing const.")
            lower_bound = max(lower_bound, const)
            const = (lower_bound + upper_bound) / 2 if upper_bound < 1e9 else const * 10
    
    return out_best_attack, out_bestscore

def generate_data(test_loader, targeted, samples, start, num_classes):
    """Generates a batch of inputs and targets for the attack."""
    inputs, targets = [], []
    count = 0
    for i, (data, label) in enumerate(test_loader):
        if i < start: continue
        if count >= samples: break
        
        if targeted:
            for j in range(num_classes):
                if j == label.item(): continue
                inputs.append(data[0].numpy())
                targets.append(np.eye(num_classes)[j])
        else:
            inputs.append(data[0].numpy())
            targets.append(np.eye(num_classes)[label.item()])
        count += 1
    return np.array(inputs), np.array(targets)

def attack_main(inputs, targets, model, device, hparams):
    """Main attack loop that iterates over all input samples."""
    adv_images = []
    print(f'Starting attack on {len(inputs)} samples...')
    for i in range(len(inputs)):
        print(f"\n--- Attacking sample {i+1}/{len(inputs)} ---")
        input_single = np.expand_dims(inputs[i], 0)
        target_single = np.expand_dims(targets[i], 0)
        
        adv_img, score = l2_attack(input_single, target_single, model, device, hparams)
        adv_images.append(adv_img)
    
    # All adv_img are (C, H, W), so stacking creates (N, C, H, W) correctly.
    return np.stack(adv_images)

if __name__ == '__main__':
    # =================================================================
    #                 CONFIGURATION PARAMETERS
    # =================================================================
    HPARAMS = {
        "dataset": "cifar10",  # "cifar10" or "mnist"
        "targeted": True,
        "solver": "adam",     # "adam" or "newton"
        "use_log": True,      # Use log_softmax for loss
        "use_tanh": True,     # Use tanh projection for perturbation
        "samples_to_attack": 2, 
        "start_from_sample": 0, 
        "batch_size": 128,      # For gradient estimation
        "max_iter": 1000,       # Iterations per binary search step
        "binary_search_steps": 5,
        "const": 0.01,          # Initial trade-off constant
        "confidence": 0,        # Attack confidence
        "step_size": 0.01       # Learning rate for the optimizer
    }
    
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # --- Model and Data Loading ---
    if HPARAMS['dataset'] == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        model = MNIST().to(device)
        model.load_state_dict(torch.load('./models/mnist_model.pt', map_location=device))
        num_classes = 10
        classes = [str(i) for i in range(10)]
    elif HPARAMS['dataset'] == 'cifar10':
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

    inputs, targets = generate_data(test_loader, HPARAMS['targeted'], HPARAMS['samples_to_attack'], HPARAMS['start_from_sample'], num_classes)
    print(f"Generated {len(inputs)} attack instances.")

    # --- Run Attack ---
    timestart = time.time()
    adv = attack_main(inputs, targets, model, device, HPARAMS)
    timeend = time.time()
    print(f"\nAttack finished. Took {(timeend - timestart) / 60.0:.2f} mins for {len(inputs)} samples.")

    # --- Evaluation ---
    assert adv.shape == inputs.shape, "Shape mismatch after attack!"
    inputs_tensor = torch.from_numpy(inputs).to(device)
    adv_tensor = torch.from_numpy(adv).to(device)

    with torch.no_grad():
        original_logits = model(inputs_tensor)
        adv_logits = model(adv_tensor)
            
    original_class = torch.argmax(original_logits, 1).cpu().numpy()
    adv_class = torch.argmax(adv_logits, 1).cpu().numpy()
    target_class = np.argmax(targets, 1)
    
    if HPARAMS['targeted']:
        success_indices = (adv_class == target_class)
        success_rate = np.mean(success_indices)
    else:
        success_indices = (adv_class != original_class)
        success_rate = np.mean(success_indices)
    
    print("\n--- Results ---")
    print(f"Original Labels:      {original_class}")
    print(f"Adversarial Labels:   {adv_class}")
    if HPARAMS['targeted']: print(f"Target Labels:        {target_class}")
    print(f"Attack Success Rate:  {success_rate * 100.0:.2f}%")

    # Calculate distortion only for successful attacks
    if np.sum(success_indices) > 0:
        distortion = np.mean(np.sum((adv[success_indices] - inputs[success_indices])**2, axis=(1,2,3))**0.5)
        print(f"Avg L2 distortion of successful attacks: {distortion:.4f}")

    # --- Visualization ---
    plt.figure(figsize=(10, 10))
    num_to_show = min(len(adv), 25)
    for i in range(num_to_show):
        plt.subplot(5, 5, i + 1)
        plt.xticks([], []); plt.yticks([], [])
        
        orig_label, adv_label = classes[original_class[i]], classes[adv_class[i]]
        plt.title(f"{orig_label} -> {adv_label}", fontsize=9)
        
        # De-normalize image from [-0.5, 0.5] to [0, 1] for plotting
        img = np.clip(adv[i] + 0.5, 0, 1)
        if HPARAMS['dataset'] == 'mnist':
            plt.imshow(img.squeeze(), cmap="gray")
        else: # cifar10
            plt.imshow(np.transpose(img, (1, 2, 0)))

    plt.tight_layout()
    filename = f"{HPARAMS['solver']}_{'targeted' if HPARAMS['targeted'] else 'untargeted'}_{HPARAMS['dataset']}_modern.png"
    plt.savefig(filename)
    print(f"\nSaved visualization to {filename}")
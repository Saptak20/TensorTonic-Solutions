import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    with torch.no_grad():
        inputs = inputs.detach()
        targets = targets.detach()
        weights = [weight.detach().clone() for weight in weights]
        biases = [bias.detach().clone() for bias in biases]

        def forward(current_weights, current_biases):
            outputs = [inputs]
            for weight, bias in zip(current_weights, current_biases):
                layer_output = torch.tanh(outputs[-1] @ weight.T + bias)
                outputs.append(layer_output)
            return outputs

        loss_history = []

        for _ in range(steps):
            outputs = forward(weights, biases)
            predictions = outputs[-1][:, 0]
            loss = ((predictions - targets) ** 2).sum()
            loss_history.append(loss)

            gradient_from_above = (2 * (predictions - targets)).unsqueeze(1)

            weight_gradients = [None] * len(weights)
            bias_gradients = [None] * len(biases)

            for layer in range(len(weights) - 1, -1, -1):
                layer_output = outputs[layer + 1]
                previous_output = outputs[layer]

                pre_activation_gradient = gradient_from_above * (1 - layer_output ** 2)

                weight_gradients[layer] = pre_activation_gradient.T @ previous_output
                bias_gradients[layer] = pre_activation_gradient.sum(dim=0)

                gradient_from_above = pre_activation_gradient @ weights[layer]

            for layer in range(len(weights)):
                weights[layer] = weights[layer] - learning_rate * weight_gradients[layer]
                biases[layer] = biases[layer] - learning_rate * bias_gradients[layer]

        final_outputs = forward(weights, biases)
        final_predictions = final_outputs[-1][:, 0]
        final_loss = ((final_predictions - targets) ** 2).sum()

    return final_predictions, final_loss, weights, biases, loss_history
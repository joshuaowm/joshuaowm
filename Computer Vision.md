### Exam: 29.01 - Written

____________________________________________________
### Exam 2024
__________________________________________________
#### Question 1. 
Say you have an image classification task with 500 classes and images of size 16x16. (1) Construct a convolutional neural network with three convolutional layers and one final linear layer, suitable for this task. For each layer, specify the layer configuration (e.g. kernel size, padding, stride, etc.) as well as input and output dimensions (and channels). (2) What would you do if you already have a *pre-trained* version of this network (trained on 1000 classes)?

_Answer:_  

(1) 
**Formula for finding output image size:** 
$Output Size = (Input Size - Kernel Size + 2 * Padding) / Stride + 1$
e.g. $Output Size = (16 - 3 + 2 * 1) / 1 + 1 = 16$

**Architecture**
- **1st Convolutional Layer:**
    - **Input dimensions:**  16×16×3, output channels: 32, kernel size: 3, stride 1, padding 1.
    - **Output dimensions:** 16×16×32.
- **ReLU**-activation function.
- **2nd Convolutional Layer:**
    - **Input dimensions:** 16×16×32, output channels: 64, kernel size: 3, stride 1, padding 1.
    - **Output dimensions:** 16×16×64.
- **ReLU**-activation function.
- **3rd Convolutional Layer:** 
    - **Input dimensions:** 16×16×64, output channels: 128, kernel size 3, stride 1, padding 1.
    - **Output dimensions:** 16×16×128
- **ReLU**-activation function.
- **Linear Layer:**
    - Flatten the output to 16×16×128=32768
    - Linear layer: 32768→500

(2)

The **Linear Layer** output has to be modified from 1000 to 500, and the model needs to be trained on the dataset.
________________________________________________________
#### Question 2.
Assume that you have a linear layer (without bias), implementing *f* : $R^2$ → $R^3$, $R^2$ ∋ x → Ax, with weight matrix 

$A=\begin{pmatrix}6&4\cr2&8\cr4&2\end{pmatrix}$ 

Fix $x=[1,2]^T$ as input with label $y=2$ (out of {0, 1, 2}) and write down the cross-entropy loss expression for this input. Do not calculate the loss value, just write down the expression with the correct values.

*Answer:* 

**Theory:** 
*Linear Transformation* performs the operation Ax. This means the input vector x is multiplied by the weight matrix A to produce the output vector.
*Input and Label*. You're given a specific vector $x=[1,2]^T$ and its true label $y=2$.
*Cross-Entropy Loss* is a <u>loss function</u> that measures the difference between the predicted probability distribution over the classes and the true probability distribution (e.g. $[0,1,0]$).

**Formula:**
Given:
* Input vector: $x = [1, 2]^T$
* Weight matrix: $A = [[6, 4], [2, 8], [4, 2]]$
* True label: $y = 2$

1. **Calculate the output of the linear layer (z):**

$z = Ax = [[6, 4], [2, 8], [4, 2]] * [1, 2]^T = [14, 18, 8]^T$

2. **Apply the softmax function to get the predicted probabilities (p):**

$p_i = exp(z_i) / sum(exp(z_j))$  for all $j=1$ to $3$

$p_1 = exp(14) / (exp(14) + exp(18) + exp(8))$
$p_2 = exp(18) / (exp(14) + exp(18) + exp(8))$
$p_3 = exp(8) / (exp(14) + exp(18) + exp(8))$

3. **Calculate the cross-entropy loss (L):**

$L = -log(p_y)$

Since $y = 2$:

$L = -log(p_2)$

Substituting the value of $p_2$:

$L = -log(exp(18) / (exp(14) + exp(18) + exp(8)))$
______________________________________
#### Question 3.
Provide the expression for the update rule in *mini batch stochastic gradient descent (SGD)* and explain each part. Why would one want to use *momentum*?

*Answer:* 

$\theta^{(t+1)}=\theta^{(t)}-\eta\displaystyle\sum_{b=1}^B\nabla_\theta l_{i(b)}(\theta^{(t)})$ 

where $\theta^{(t+1)}$ is the new parameter, $\theta^{(t)}$ is the current parameter, $\eta$ is the learning rate, $\nabla_\theta l_{i(b)}(\theta^{(t)})$ is the gradient descent with respect to $\theta$ for the selected sample. *Mini batch SGD* takes randomly selected mini batches of parameters and finds the loss with gradient, and updates the parameters. 

Momentum changes how big of a step does update function take by introducing a $\gamma$ - momentum parameter or 'velocity'.  It controls how much of the previous velocity is retained. A higher gamma means more "momentum."

$v^{(t+1)} = γ * v^{(t)} + η * (1/B) * Σ_{b=1}^B ∇_θ l_{i(b)}(θ^{(t)})$
$θ^{(t+1)} = θ^{(t)} - v^{(t+1)}$
_________________
#### Question 4.
Say the output of a 2D convolution layer produces the feature maps shown below, with W=H=12 and C=40 (i.e., 40 channels).

![[Pasted image 20250128112332.png]]

Specify the configuration (i.e., number of input/output channels, kernel size, padding, stride) of one 2D convolution layer with a minimal number of parameters to obtain one output feature map of size W/2 × H/2. Ignore any biases.

*Answer:*

**Formula:**
$Output Size = (Input Size - Kernel Size + 2 * Padding) / Stride + 1$
$Number Of Parameters = (Kernel Height * Kernel Width * Input Channels * Output Channels) + OutputChannels(if bias = True)$

**Calculation:**

(1)
$Output Size = (12 - 7 + 2 * 0) / 1 + 1 = 6$
$Number of Parameters = 7 * 7 * 40 * 1 = 1960$

(2)
$Output Size = (12 - 2 + 2 * 0) / 2 + 1 = 6$
$Number of Parameters = 2 * 2 * 40 * 1 = 160$

Parameters of convolutional layer would be: input channels = 40, output channels = 1, kernel size = 2, padding = 0, stride = 2.
_______________
#### Question 5.
Look at the following code snippet:
```python
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = 3*x**2 + 4*y*x.detach()
z.backward()
print(x.grad, y.grad)
```
What will be the output of the print statement in the last line. *Hint: note the use of detach()*.

*Answer:*

$\frac{dz}{dx} = 2*3*x = 2*3*2 = 12$
$\frac{dz}{dy} = 4*x = 4*2 = 8$

The output would be 12, 8
_____________________
### Exam 2023
_________________________
#### Question 1.
Look at the following naive implementation of the softmax function scaled by a temperature parameter k:
```python
def softmax(x,k=1.):
	return torch.exp(x/k)/torch.sum(torch.exp(x/k))
```
Now consider the following code snippet:
```python
x0 = torch.tensor(1.0,2.0,3.0,5.0,5.0)
print(softmax(x0,1e-3))

>> tensor([nan,nan,nan,nan,nan])
```
Please (1) explain what happens here and (2) provide the output vector (approximately) that you would have expected, from a purely mathematical point of view (remember that the softmax actually seeks to approximate).

*Answer:*
(1)

(2)
$p_1 = exp(1) / exp(1) + exp(2) + exp(3) + exp(5) + exp(5)$ 
$p_2 = exp(2) / exp(1) + exp(2) + exp(3) + exp(5) + exp(5)$ 
$p_3 = exp(3) / exp(1) + exp(2) + exp(3) + exp(5) + exp(5)$ 
$p_4 = exp(5) / exp(1) + exp(2) + exp(3) + exp(5) + exp(5)$ 
$p_5 = exp(5) / exp(1) + exp(2) + exp(3) + exp(5) + exp(5)$ 

(1-1)/(5-1) = 0.05
(2-1)/(5-1) = 0.1
(3-1)/(5-1) = 0.2
(5-1)/(5-1) = 0.4
(5-1)/(5-1) = 0.4

Expected approximate output: ()
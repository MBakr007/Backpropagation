# Back propagation Neural Networks

In both Project, you are reading the input (x vector) and the output (y vector) from a file.

### Below is the structure of the input file:
#### 1) First line: M, L, N where M is number of Input Nodes, L is number of Hidden Nodes and N is number of Output Nodes.
#### 2) Second line: K, the number of training examples, each line has length M+N values, first M values are X vector and last N values are output values.
#### 3) K lines follow as described.

### An example of input file:
3 2 2

3

1 1 1.5   2  2

-1 2.25 0.5     -0.5 1.2

1 1 1     1   2

#### Above is a file that describes:
1) Network with 3 input nodes, 2 hidden and 2 output.
2) Training is 3 examples.
3) Second example has training example X [1 1 1.5] and output.

##### Normalization is done by computing, for each numeric  x-data column value v, v' = (v - mean) / std dev. This  technique is sometimes called Gaussian normalization.



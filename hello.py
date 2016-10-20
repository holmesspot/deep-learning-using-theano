import theano
import theano.tensor as T 
from theano import shared
import numpy as np 

rng = np.random

N = 400
feats = 784

D = (rng.rand(N, feats), rng.randint(0, 2, N))
training_steps = 100

x = T.dmatrix('x')
y = T.bvector('y')
w_init=rng.randn(feats)
w = shared(w_init, name='w')
b = shared(0)

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xcent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
cost = xcent.mean()
gw, gb = T.grad(cost, [w, b])

train = theano.function(
			inputs = [x, y],
			outputs = [prediction, xcent],
			updates = ((w, w - 0.1 * gw), (b, b - 0.1 * gb))
			)

for i in range(training_steps):
	pred, xcen = train(D[0], D[1])

print('final_model')
print('w',w.get_value)
print('b',b.get_value)

print('----------------')

# Declare Theano symbolic variables
x123 = T.dmatrix("x")
y123 = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w123 = theano.shared(w_init, name="w")

# initialize the bias term
b123 = theano.shared(0., name="b")



# Construct Theano expression graph
p_1123 = 1 / (1 + T.exp(-T.dot(x123, w123) - b123))   # Probability that target = 1
prediction123 = p_1123 > 0.5                    # The prediction thresholded
xent123 = -y123 * T.log(p_1123) - (1-y123) * T.log(1-p_1123) # Cross-entropy loss function
cost123 = xent123.mean() + 0.01 * (w123 ** 2).sum()# The cost to minimize
gw123, gb123 = T.grad(cost123, [w123, b123])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train123 = theano.function(
          inputs=[x123,y123],
          outputs=[prediction123, xent123],
          updates=((w123, w123 - 0.1 * gw123), (b123, b123 - 0.1 * gb123)))
#predict123 = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train123(D[0], D[1])

print("Final model:")
print(w123.get_value())
print(b123.get_value())






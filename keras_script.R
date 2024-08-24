# Install keras package
#install.packages("keras")

# Load libraries
library(tidyverse)
library(keras)

# Install TensorFlow
#install_keras()

# Recognizing handwritten digits from the MNIST data set, which consists of
# 28x28 grayscale images of handwritten digits

# Load MNIST data set
mnist <- dataset_mnist()

# Train data set
x_train <- mnist$train$x
y_train <- mnist$train$y

# Test data set
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape width and height into a single dimension (28x28 images are flattened
# into length 784 vectors)
x_train <- array_reshape(x = x_train, dim = c(nrow(x_train), order = 28*28))
x_test <- array_reshape(x = x_test, dim = c(nrow(x_test), order = 28*28))

# Rescale to convert the grayscale values from integers ranging between 0 to 255 
# into floating point values ranging between 0 and 1
x_train <- x_train / 255
x_test <- x_test / 255

# y data is an integer vector with values ranging from 0 to 9. To prepare this 
# data for training we one-hot encode the vectors into binary class matrices using
# the Keras to_categorical() function
y_train <- to_categorical(y = y_train, num_classes = 10)
y_test <- to_categorical(y = y_test, num_classes = 10)

# Defining the model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(name = "tf.keras.optimizers.legacy.RMSprop"),
  metrics = c('accuracy')
)

# Training and evaluating
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


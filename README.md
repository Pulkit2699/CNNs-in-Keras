# CNNs-in-Keras

Using TensorFlow

Functions:
get_dataset(training=True) — takes an optional boolean argument and returns the data as described below
build_model() — takes no arguments and returns an untrained neural network as specified below
train_model(model, train_img, train_lab, test_img, test_lab, T) — takes the model produced by the previous function and the images and labels produced by the first function and trains the data for T epochs; does not return anything
predict_label(model, images, index) — takes the trained model and test images, and prints the top 3 most likely labels for the image at the given index, along with their probabilities

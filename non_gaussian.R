# Load necessary libraries
library(MASS)
library(Matrix)
library(mvtnorm)

# Load the filtered dataset
data <- read.csv("star_classification.csv")

# Convert the class variable to numeric
data$class <- as.numeric(as.factor(data$class))  # Convert class labels to numeric
class_labels <- levels(as.factor(data$class))

# Extract predictors and response
X <- as.matrix(data[, c("u", "g", "r", "i")])    # Predictor matrix
y <- data$class                                  # Response variable

# Parameters for Gibbs Sampling
n <- nrow(X)
p <- ncol(X)
K <- length(unique(y))   # Number of classes
iterations <- 5000        # Reduced number of iterations for speed

# Initialize parameters
beta <- matrix(0, nrow = p, ncol = K)            # Regression coefficients
beta_samples <- array(0, dim = c(iterations, p, K))  # Store beta samples

# Prior parameters
beta_prior_mean <- rep(0, p)
beta_prior_cov_inv <- diag(1, p)                 # Inverse of covariance matrix

# Precompute XtX for efficiency
XtX <- crossprod(X)

# Gibbs Sampling Loop
for (iter in 1:iterations) {
  # Update beta for each class
  for (k in 1:K) {
    y_k <- as.numeric(y == k)                   # Binary response for class k
    cov_k <- solve(XtX + beta_prior_cov_inv)    # Posterior covariance
    mean_k <- cov_k %*% crossprod(X, y_k)       # Posterior mean
    beta[, k] <- mvrnorm(1, mean_k, cov_k)      # Sample beta_k
  }
  
  # Store beta samples
  beta_samples[iter, , ] <- beta
}

# Predict classes using the final beta samples
final_beta <- apply(beta_samples, c(2, 3), mean)  # Use the average beta over iterations
logits <- X %*% final_beta
softmax <- function(x) exp(x) / sum(exp(x))       # Define softmax function
predicted_probs <- t(apply(logits, 1, softmax))   # Calculate probabilities
predicted_classes <- apply(predicted_probs, 1, which.max)

# Create a Data Frame with Predictions and Probabilities
results <- data.frame(
  Actual = y,
  Predicted = predicted_classes,
  predicted_probs
)

results
# Confusion Matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = y)
print(confusion_matrix)

# Accuracy Calculation
correct_predictions <- sum(diag(confusion_matrix))  # Correct predictions
total_predictions <- sum(confusion_matrix)          # Total predictions
accuracy <- correct_predictions / total_predictions
cat("Accuracy of the model:", accuracy, "\n")

# Visualize Class Probabilities for One Class (Example: Class 1)
library(ggplot2)
prob_df <- data.frame(
  Actual = as.factor(y),
  Prob = predicted_probs[, 1]  # Probability for Class 1
)

ggplot(prob_df, aes(x = Actual, y = Prob)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Posterior Probabilities for Class 1", x = "Actual Class", y = "Probability") +
  theme_minimal()

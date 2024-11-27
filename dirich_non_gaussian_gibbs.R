# Load necessary libraries
library(MASS)
library(Matrix)
library(caret)

# Load the dataset
data <- read.csv("star_classification.csv")

# Convert the class variable to numeric
data$class <- as.numeric(as.factor(data$class))  # Convert class labels to numeric
class_labels <- levels(as.factor(data$class))

# Feature Engineering: Add Polynomial and Interaction Terms
data$u2 <- data$u^2
data$g2 <- data$g^2
data$r2 <- data$r^2
data$i2 <- data$i^2
data$u_g <- data$u * data$g
data$u_r <- data$u * data$r
data$u_i <- data$u * data$i
data$g_r <- data$g * data$r
data$g_i <- data$g * data$i
data$r_i <- data$r * data$i

# Scale predictors
X <- scale(as.matrix(data[, c("u", "g", "r", "i", "u2", "g2", "r2", "i2", 
                              "u_g", "u_r", "u_i", "g_r", "g_i", "r_i")]))
y <- data$class                                       # Response variable

# Parameters for Gibbs Sampling
n <- nrow(X)
p <- ncol(X)
K <- length(unique(y))       # Number of classes
iterations <- 5000           # Increased iterations for convergence
burn_in <- 2500              # Burn-in period

# Initialize parameters
beta <- matrix(0, nrow = p, ncol = K)            # Regression coefficients
beta_samples <- array(0, dim = c(iterations, p, K))  # Store beta samples

# Dirichlet Prior for Class Probabilities
alpha_dirichlet <- rep(2, K)                     # Stronger concentration prior
theta <- rep(1 / K, K)                           # Initial class probabilities

# Gaussian Prior for Coefficients
beta_prior_mean <- rep(0, p)
beta_prior_cov_inv <- diag(0.01, p)              # Tighter regularization

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
  
  # Update class probabilities (Dirichlet prior)
  counts <- table(factor(y, levels = 1:K)) + 1  # Add 1 for smoothing
  theta <- rdirichlet(1, alpha_dirichlet + as.numeric(counts))
  
  # Store beta samples
  beta_samples[iter, , ] <- beta
}

# Discard burn-in samples and average the posterior samples
final_beta <- apply(beta_samples[(burn_in + 1):iterations, , ], c(2, 3), mean)

# Predict classes using the final beta samples
logits <- X %*% final_beta
softmax <- function(x) exp(x) / sum(exp(x))       # Define softmax function
predicted_probs <- t(apply(logits, 1, softmax))   # Calculate probabilities
predicted_classes <- apply(predicted_probs, 1, which.max)

# Evaluate Model
confusion_matrix <- table(Predicted = predicted_classes, Actual = y)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Print Results
cat("Confusion Matrix:\n")
print(confusion_matrix)
cat("Accuracy of the Model:", accuracy, "\n")

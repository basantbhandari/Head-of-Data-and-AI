<!-- notion-id: 3d4674182854806e85bbcd2540b07ef9 -->
# Maths

Maths interview questions for ML: linear algebra fundamentals (eigenvalues and eigenvectors).

### Q1. Eigenvalues and Eigenvectors

**Eigenvalues and Eigenvectors**

- **Eigenvector:** A non-zero vector whose direction does not change when a matrix transformation is applied to it. Only its magnitude may change.
- **Eigenvalue:** The scalar that tells how much the corresponding eigenvector is stretched or compressed by the transformation.
- They are defined by: **A v = λ v**
  - `A` = square matrix
  - `v` = eigenvector
  - `λ` = eigenvalue
- **Eigenvalues** are found by solving the characteristic equation:**det(A − λI) = 0**
- **Eigenvectors** are then found by solving:**(A − λI)v = 0**
- **In ML**, eigenvalues and eigenvectors are important in:
  - PCA and dimensionality reduction
  - Covariance matrix analysis
  - Spectral clustering
  - Graph neural networks
  - Stability analysis
  - Optimization and numerical methods

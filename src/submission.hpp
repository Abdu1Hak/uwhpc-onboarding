#pragma once
#include <cstddef>
#include <vector> 
// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> flat_data_; // contiguous memory 

public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), flat_data_(rows * cols, 0.0){}; // zero init
  // 2. assinging member variables before running constructor is more efficient
  
  // read-write operator + & means reference to i,j cell value
  double& operator()(std::size_t i, std::size_t j){
    return flat_data_[i * cols_ + j]; 
  } 

  double  operator()(std::size_t i, std::size_t j) const {
    return flat_data_[i * cols_ + j]; 
  }

  // Expose dimensions so apply_stencil can loop over them
  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.

//inline allows the definition to exist translation units without causing a multiple-definition error
inline void apply_stencil(const Grid& old_grid, Grid& new_grid){ 
  const std::size_t R = old_grid.rows(); 
  const std::size_t C = old_grid.cols(); 

  // Copy boundary rows - iterator j for all column indexes, copy the outer boundary of the old
  for (std::size_t j = 0; j<C; ++j){
    new_grid(0, j) = old_grid(0, j); // top row 
    new_grid(R-1, j) = old_grid(R-1, j); // bottom row
  }
  for (std::size_t i=0; i<R; i++){
    new_grid(i, 0) = old_grid(i, 0); // left col
    new_grid(i, C-1) = old_grid(i, C-1); // right col

  }

  // Apply the simplified stencil to each interior cell 
  // nested for-loop in design 1 
  for (std::size_t i = 1; i<R-1; i++){
    for (std::size_t j = 1; j<C-1; j++){
      new_grid(i, j) = 0.5 * old_grid(i, j) + 0.125 * old_grid(i-1, j) +
                    0.125 * old_grid(i+1, j) + 0.125 * old_grid(i, j-1) + 
                    0.125 * old_grid(i, j+1); 
    }
  }

};

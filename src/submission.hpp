#pragma once
#include <cstddef>
#include <vector>     
// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.

// Ownership/View
// A view is a plain struct, containing pointers + dimensions + stride fully transparent to optimizer

template<typename T> // placeholder for unsure data type for the two views
struct GridViewT { 
  T* __restrict__ data; // points directly to the Grid's vector (prevents overlap memory in old/new)
  std::size_t    rows; 
  std::size_t    cols; 
  std::size_t    stride; // row width after padding (SIMD)

  T& operator() (std::size_t i, std::size_t j) const { 
    return data[i * stride + j];
  }
}; 

using GridView = GridViewT<double>; // read-write 
using ConstGridView = GridViewT<const double>; // read-only

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double> flat_data_; // contiguous memory 
  
public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), stride_(cols), flat_data_(rows * cols, 0.0){}; // zero init
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

  GridView view() { return {flat_data_.data(), rows_, cols_, stride_};}
  ConstGridView view() const {return {flat_data_.data(), rows_, cols_, stride_}; }
  
};  



//inline allows the definition to exist translation units without causing a multiple-definition error
inline void apply_stencil(ConstGridView old_v, GridView new_v){ 
  const auto R = old_v.rows;
  const auto C = old_v.cols;

  // Copy boundary rows - iterator j for all column indexes, copy the outer boundary of the old
  for (std::size_t j = 0; j<C; ++j){
    new_v(0,   j) = old_v(0,   j); // top row
    new_v(R-1, j) = old_v(R-1, j); // bottom row
  }

  // 2B - Cache Friendly Boundary Copying
  // 2B use OpenMP multi-thread for row iteration, and SIMD for multi iterations (AVX2)
  #pragma omp parallel for schedule(static) // split the iteration across openmp threads
  for (std::size_t i=1; i<R-1; i++){
    // Left and Right boundary - same row means same cache lines as stencil (less memory bandwidth)
    // The idea is that when row i's cache line are loaded, the stencil runs on that row hitting the same cache line. 
    new_v(i, 0) = old_v(i, 0); // left col
    new_v(i, C-1) = old_v(i, C-1); // right col

    #pragma omp simd // vectorize the loop and multi iterate using SIMD registers
    for (std::size_t j = 1; j<C-1; ++j){
      new_v(i, j) = 0.5   * old_v(i,   j  )
                    + 0.125 * old_v(i-1, j  )
                    + 0.125 * old_v(i+1, j  )
                    + 0.125 * old_v(i,   j-1)
                    + 0.125 * old_v(i,   j+1);
    }
  }
}

inline void apply_stencil(const Grid& old_g, Grid& new_g) {
  apply_stencil(old_g.view(), new_g.view());
}
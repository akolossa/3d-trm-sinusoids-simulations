#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Constants from the paper
constexpr float LIVER_LENGTH_CM = 1.0f;          // 1 cm^3 liver volume
constexpr int JUNCTION_SPACING_UM = 50;          // 50 micromm between sinusoid junctions
constexpr int JUNCTIONS_PER_DIM = static_cast<int>(LIVER_LENGTH_CM * 1e4f / JUNCTION_SPACING_UM) + 1;  // 201
constexpr int TOTAL_JUNCTIONS = JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM;  // ~8.12 million
constexpr float TIME_STEP_MIN = 12.0f / M_PIf32;  // around 3.82 minutes per step (from paper)
constexpr int BURN_IN_STEPS = 1000;               

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel for initializing random number generators
__global__ void init_rng_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + idx, 0, 0, &state[idx]);
}

// Kernel for random walk step (burn-in phase)
__global__ void burn_in_kernel(int* tcells, curandState* states, int num_tcells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tcells) return;

    curandState local_state = states[idx];
    
    for (int step = 0; step < BURN_IN_STEPS; step++) {
        // Each T cell is represented by 3 consecutive integers (x,y,z)
        int* pos = &tcells[idx * 3];
        
        // Random axis (0=x, 1=y, 2=z)
        int axis = curand(&local_state) % 3;
        // Random direction (-1 or 1)
        int direction = (curand(&local_state) % 2) * 2 - 1;
        
        // Update position
        pos[axis] += direction;
        // Apply periodic boundary conditions
        pos[axis] = (pos[axis] + JUNCTIONS_PER_DIM) % JUNCTIONS_PER_DIM;
    }
    
    states[idx] = local_state;
}

// Kernel for simulation step (with coverage tracking)
__global__ void simulation_step_kernel(int* tcells, bool* visited, curandState* states, 
                                      int num_tcells, int* coverage_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tcells) return;

    curandState local_state = states[idx];
    int* pos = &tcells[idx * 3];
    
    // Random axis (0=x, 1=y, 2=z)
    int axis = curand(&local_state) % 3;
    // Random direction (-1 or 1)
    int direction = (curand(&local_state) % 2) * 2 - 1;
    
    // Update position
    pos[axis] += direction;
    // Apply periodic boundary conditions
    pos[axis] = (pos[axis] + JUNCTIONS_PER_DIM) % JUNCTIONS_PER_DIM;
    
    // Mark visited junction
    int x = pos[0];
    int y = pos[1];
    int z = pos[2];
    
    // Calculate linear index for visited array
    int junction_idx = x + y * JUNCTIONS_PER_DIM + z * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM;
    
    // Atomic operation to mark as visited and count if it's newly visited
    bool already_visited = atomicExch(&visited[junction_idx], true);
    if (!already_visited) {
        atomicAdd(coverage_count, 1);
    }
    
    states[idx] = local_state;
}

class LiverModelDiscrete {
public:
    LiverModelDiscrete() : visited(JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM, false) {
        // Initialize CUDA random number generators
        CHECK_CUDA_ERROR(cudaMalloc(&d_states, MAX_TCELLS * sizeof(curandState)));
        init_rng_kernel<<<(MAX_TCELLS + 255) / 256, 256>>>(d_states, std::chrono::system_clock::now().time_since_epoch().count());
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Allocate device memory for coverage count
        CHECK_CUDA_ERROR(cudaMalloc(&d_coverage_count, sizeof(int)));
    }
    
    ~LiverModelDiscrete() {
        cudaFree(d_states);
        cudaFree(d_tcells);
        cudaFree(d_visited);
        cudaFree(d_coverage_count);
    }
    
    void initialize_tcells(int num_tcells) {
        this->num_tcells = num_tcells;
        
        // Allocate device memory for T cells
        CHECK_CUDA_ERROR(cudaMalloc(&d_tcells, num_tcells * 3 * sizeof(int)));
        
        // Initialize T cells at random positions on host
        thrust::host_vector<int> h_tcells(num_tcells * 3);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, JUNCTIONS_PER_DIM - 1);
        
        for (int i = 0; i < num_tcells * 3; i++) {
            h_tcells[i] = dis(gen);
        }
        
        // Copy to device
        thrust::device_vector<int> d_tcells_vec = h_tcells;
        int* raw_ptr = thrust::raw_pointer_cast(d_tcells_vec.data());
        CHECK_CUDA_ERROR(cudaMemcpy(d_tcells, raw_ptr, num_tcells * 3 * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    void burn_in() {
        int block_size = 256;
        int grid_size = (num_tcells + block_size - 1) / block_size;
        
        burn_in_kernel<<<grid_size, block_size>>>(d_tcells, d_states, num_tcells);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    
    std::pair<std::vector<float>, std::vector<float>> simulate_coverage(int target_coverage_percent = 90, int max_steps = 1000) {
        // Reset visited array
        thrust::fill(visited.begin(), visited.end(), false);
        
        // Allocate and copy visited array to device
        CHECK_CUDA_ERROR(cudaMalloc(&d_visited, visited.size() * sizeof(bool)));
        bool* h_visited = visited.data();
        CHECK_CUDA_ERROR(cudaMemcpy(d_visited, h_visited, visited.size() * sizeof(bool), cudaMemcpyHostToDevice));
        
        std::vector<float> coverage_history;
        std::vector<float> time_points;
        float total_time_min = 0.0f;
        
        int block_size = 256;
        int grid_size = (num_tcells + block_size - 1) / block_size;
        
        for (int step = 0; step < max_steps; step++) {
            // Reset coverage count
            int zero = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(d_coverage_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
            
            // Run simulation step
            simulation_step_kernel<<<grid_size, block_size>>>(d_tcells, d_visited, d_states, num_tcells, d_coverage_count);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            // Get coverage count
            int coverage_count;
            CHECK_CUDA_ERROR(cudaMemcpy(&coverage_count, d_coverage_count, sizeof(int), cudaMemcpyDeviceToHost));
            
            // Calculate coverage percentage
            float coverage = (static_cast<float>(coverage_count) / TOTAL_JUNCTIONS) * 100.0f;
            coverage_history.push_back(coverage);
            
            total_time_min += TIME_STEP_MIN;
            time_points.push_back(total_time_min);
            
            // Print progress
            if (step % 10 == 0) {
                std::cout << "Step " << step << ": " << coverage << "% coverage" << std::endl;
            }
            
            // Early stopping if target reached
            if (coverage >= target_coverage_percent) {
                break;
            }
        }
        
        return {coverage_history, time_points};
    }
    
    void plot_results(const std::vector<float>& coverage_history, const std::vector<float>& time_points) {
        // Simple ASCII plot (for actual plotting, consider using a library like gnuplot or matplotlib-cpp)
        std::cout << "\nCoverage over time:\n";
        std::cout << "Time (hours)\tCoverage (%)\n";
        for (size_t i = 0; i < coverage_history.size(); i++) {
            std::cout << std::fixed << std::setprecision(2) 
                      << time_points[i] / 60.0f << "\t\t" 
                      << coverage_history[i] << "\n";
        }
        
        // Write data to file for external plotting
        std::ofstream outfile("coverage_data.csv");
        outfile << "Time(hours),Coverage(%)\n";
        for (size_t i = 0; i < coverage_history.size(); i++) {
            outfile << time_points[i] / 60.0f << "," << coverage_history[i] << "\n";
        }
        outfile.close();
        std::cout << "\nData written to coverage_data.csv for plotting\n";
    }

private:
    static constexpr int MAX_TCELLS = 10000000;  // Maximum number of T cells we'll support
    
    int num_tcells;
    thrust::host_vector<bool> visited;
    
    // Device pointers
    int* d_tcells = nullptr;
    bool* d_visited = nullptr;
    curandState* d_states = nullptr;
    int* d_coverage_count = nullptr;
};

int main() {
    std::cout << "Liver Model Simulation (GPU Accelerated)\n";
    std::cout << "Grid size: " << JUNCTIONS_PER_DIM << "x" << JUNCTIONS_PER_DIM << "x" << JUNCTIONS_PER_DIM << "\n";
    std::cout << "Total junctions: " << TOTAL_JUNCTIONS << "\n";
    std::cout << "Time step: " << TIME_STEP_MIN << " minutes\n";
    
    LiverModelDiscrete model;
    
    int num_tcells = 10000000;  // 10 million T cells
    std::cout << "\nInitializing " << num_tcells << " T cells...\n";
    model.initialize_tcells(num_tcells);
    
    std::cout << "Running burn-in period of " << BURN_IN_STEPS << " steps...\n";
    auto start = std::chrono::high_resolution_clock::now();
    model.burn_in();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Burn-in completed in " << elapsed.count() << " seconds\n";
    
    std::cout << "\nStarting simulation...\n";
    start = std::chrono::high_resolution_clock::now();
    auto [coverage, times] = model.simulate_coverage(90, 1000);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds\n";
    
    // Print results
    float final_time_hours = times.back() / 60.0f;
    std::cout << "\nTime to reach 90% coverage: " << final_time_hours << " hours\n";
    std::cout << "Final coverage: " << coverage.back() << "% at step " << coverage.size() << "\n";
    
    // Plot results
    model.plot_results(coverage, times);
    
    return 0;
}
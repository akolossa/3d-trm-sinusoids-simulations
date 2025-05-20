#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <sys/stat.h>

// Constants - Adjusted as requested
constexpr float LIVER_LENGTH_CM = 0.0128f;  // 128 microns
constexpr int JUNCTION_SPACING_UM = 1;      // 1 micron spacing
constexpr int JUNCTIONS_PER_DIM = 128;       // 128x128x128 grid
constexpr int TOTAL_JUNCTIONS = JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM;
constexpr float TIME_STEP_MIN = 0.1f;        // 0.1 minute per step 
constexpr int BURN_IN_STEPS = 60;            // same burnin as gpucpm
constexpr int NUM_TCELLS = 8;                // 8 T-cells

const std::string OUTPUT_DIR = "/home/arawa/liver-3d-simulations/3D_simulations/computational_power_comparison";

class LiverModelDiscrete {
public:
    LiverModelDiscrete() : 
        visited(TOTAL_JUNCTIONS, false),
        coverage_history(),
        time_points(),
        msd_history() {
        
        // Create output directory
        struct stat st;
        if (stat(OUTPUT_DIR.c_str(), &st) != 0) {
            mkdir(OUTPUT_DIR.c_str(), 0777);
        }
    }

    void initialize_tcells() {
        tcells.resize(NUM_TCELLS * 3);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, JUNCTIONS_PER_DIM - 1);

        #pragma omp parallel for
        for (int i = 0; i < NUM_TCELLS * 3; i++) {
            tcells[i] = dis(gen);
        }
        previous_positions = tcells;
    }

    void burn_in() {
        std::random_device rd;
        #pragma omp parallel
        {
            std::mt19937 gen(rd() + omp_get_thread_num());
            std::uniform_int_distribution<> axis_dist(0, 2);
            std::uniform_int_distribution<> dir_dist(0, 1);

            #pragma omp for
            for (int i = 0; i < NUM_TCELLS; i++) {
                for (int step = 0; step < BURN_IN_STEPS; step++) {
                    int axis = axis_dist(gen);
                    int direction = dir_dist(gen) * 2 - 1;
                    tcells[i * 3 + axis] = 
                        (tcells[i * 3 + axis] + direction + JUNCTIONS_PER_DIM) % JUNCTIONS_PER_DIM;
                }
            }
        }
    }

    void simulate_coverage(int max_steps = 1000) {
        std::fill(visited.begin(), visited.end(), false);
        float total_time_min = 0.0f;  // Time tracking starts AFTER burn-in
        int total_covered = 0;

        std::random_device rd;
        for (int step = 0; step < max_steps; step++) {
            int step_covered = 0;
            float current_msD = 0.0f;

            #pragma omp parallel reduction(+:step_covered, current_msD)
            {
                std::mt19937 gen(rd() + omp_get_thread_num());
                std::uniform_int_distribution<> axis_dist(0, 2);
                std::uniform_int_distribution<> dir_dist(0, 1);

                #pragma omp for
                for (int i = 0; i < NUM_TCELLS; i++) {
                    // Random walk (1μm step)
                    int axis = axis_dist(gen);
                    int direction = dir_dist(gen) * 2 - 1;
                    int& pos = tcells[i * 3 + axis];
                    pos = (pos + direction + JUNCTIONS_PER_DIM) % JUNCTIONS_PER_DIM;

                    // Track coverage
                    int idx = tcells[i * 3] + tcells[i * 3 + 1] * JUNCTIONS_PER_DIM + 
                             tcells[i * 3 + 2] * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM;
                    if (!visited[idx]) {
                        visited[idx] = true;
                        step_covered++;
                    }

                    // Calculate MSD (in μm²)
                    int dx = tcells[i*3] - previous_positions[i*3];
                    int dy = tcells[i*3+1] - previous_positions[i*3+1];
                    int dz = tcells[i*3+2] - previous_positions[i*3+2];
                    current_msD += dx*dx + dy*dy + dz*dz;
                }
            }

            previous_positions = tcells;
            total_covered += step_covered;
            total_time_min += TIME_STEP_MIN;  

            // Store metrics
            coverage_history.push_back((static_cast<float>(total_covered) / TOTAL_JUNCTIONS) * 100.0f);
            time_points.push_back(total_time_min);
            msd_history.push_back(current_msD / NUM_TCELLS);

            if (step % 10 == 0) {
                std::cout << "Step " << step << " (Time: " << total_time_min << " min): " 
                          << coverage_history.back() << "% coverage, MSD: " 
                          << msd_history.back() << " μm²\n";
            }

            if (coverage_history.back() >= 40.0f) break;
        }
    }

    void save_results() {
        std::string coverage_file = OUTPUT_DIR + "/coverage_metrics_Discrete.csv";
        std::ofstream outfile(coverage_file);
        outfile << "Time(min),Coverage(%),MSD(um2)\n";
        for (size_t i = 0; i < coverage_history.size(); i++) {
            outfile << time_points[i] << "," << coverage_history[i] 
                   << "," << msd_history[i] << "\n";
        }
        outfile.close();
    }

private:
    std::vector<int> tcells;
    std::vector<int> previous_positions;
    std::vector<bool> visited;
    std::vector<float> coverage_history;
    std::vector<float> time_points;
    std::vector<float> msd_history;
};

int main() {
    LiverModelDiscrete model;
    
    std::cout << "Initializing " << NUM_TCELLS << " T-cells...\n";
    model.initialize_tcells();
    
    std::cout << "Running burn-in (" << BURN_IN_STEPS << " steps)...\n";
    model.burn_in();
    
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting main simulation (1 min/step)...\n";
    model.simulate_coverage();
    auto end = std::chrono::high_resolution_clock::now();
    
    model.save_results();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation completed in " << elapsed.count() << " seconds\n";
    std::cout << "Results saved to " << OUTPUT_DIR << "\n";
    
    return 0;
}
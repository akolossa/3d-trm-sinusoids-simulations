#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>

// Constants from the paper
constexpr float LIVER_LENGTH_CM = 1.0f;
constexpr int JUNCTION_SPACING_UM = 50;
constexpr int JUNCTIONS_PER_DIM = static_cast<int>(LIVER_LENGTH_CM * 1e4f / JUNCTION_SPACING_UM) + 1;
constexpr int TOTAL_JUNCTIONS = JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM;
constexpr float TIME_STEP_MIN = 12.0f / 3.14159265358979323846f;  // Fixed: Replaced M_PIf32
constexpr int BURN_IN_STEPS = 1000;

class LiverModelDiscrete {
public:
    LiverModelDiscrete() : visited(TOTAL_JUNCTIONS, false) {}

    void initialize_tcells(int num_tcells) {
        this->num_tcells = num_tcells;
        tcells.resize(num_tcells * 3);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, JUNCTIONS_PER_DIM - 1);

        #pragma omp parallel for
        for (int i = 0; i < num_tcells * 3; i++) {
            tcells[i] = dis(gen);
        }
    }

    void burn_in() {
        std::random_device rd;
        #pragma omp parallel
        {
            std::mt19937 gen(rd() + omp_get_thread_num());
            std::uniform_int_distribution<> axis_dist(0, 2);
            std::uniform_int_distribution<> dir_dist(0, 1);

            #pragma omp for
            for (int i = 0; i < num_tcells; i++) {
                for (int step = 0; step < BURN_IN_STEPS; step++) {
                    int axis = axis_dist(gen);
                    int direction = dir_dist(gen) * 2 - 1;
                    
                    int& pos = tcells[i * 3 + axis];
                    pos += direction;
                    pos = (pos + JUNCTIONS_PER_DIM) % JUNCTIONS_PER_DIM;
                }
            }
        }
    }

    std::pair<std::vector<float>, std::vector<float>> simulate_coverage(int target_coverage_percent = 40, int max_steps = 1000) {
        std::fill(visited.begin(), visited.end(), false);
        std::vector<float> coverage_history;
        std::vector<float> time_points;
        float total_time_min = 0.0f;
        int total_covered = 0;

        std::random_device rd;
        for (int step = 0; step < max_steps; step++) {
            int step_covered = 0;

            #pragma omp parallel
            {
                std::mt19937 gen(rd() + omp_get_thread_num());
                std::uniform_int_distribution<> axis_dist(0, 2);
                std::uniform_int_distribution<> dir_dist(0, 1);

                #pragma omp for reduction(+:step_covered)
                for (int i = 0; i < num_tcells; i++) {
                    int axis = axis_dist(gen);
                    int direction = dir_dist(gen) * 2 - 1;
                    
                    int& pos = tcells[i * 3 + axis];
                    pos += direction;
                    pos = (pos + JUNCTIONS_PER_DIM) % JUNCTIONS_PER_DIM;

                    int x = tcells[i * 3];
                    int y = tcells[i * 3 + 1];
                    int z = tcells[i * 3 + 2];
                    int idx = x + y * JUNCTIONS_PER_DIM + z * JUNCTIONS_PER_DIM * JUNCTIONS_PER_DIM;

                    if (!visited[idx]) {
                        visited[idx] = true;
                        step_covered++;
                    }
                }
            }

            total_covered += step_covered;
            float coverage = (static_cast<float>(total_covered) / TOTAL_JUNCTIONS) * 100.0f;
            coverage_history.push_back(coverage);
            total_time_min += TIME_STEP_MIN;
            time_points.push_back(total_time_min);

            if (step % 10 == 0) {
                std::cout << "Step " << step << ": " << coverage << "% coverage" << std::endl;
            }

            if (coverage >= target_coverage_percent) {
                break;
            }
        }

        return {coverage_history, time_points};
    }

    void plot_results(const std::vector<float>& coverage_history, const std::vector<float>& time_points) {
        std::ofstream outfile("coverage_data.csv");
        outfile << "Time(hours),Coverage(%)\n";
        for (size_t i = 0; i < coverage_history.size(); i++) {
            outfile << time_points[i] / 60.0f << "," << coverage_history[i] << "\n";
        }
        outfile.close();
        std::cout << "\nData written to coverage_data.csv\n";
    }

private:
    int num_tcells;
    std::vector<int> tcells;
    std::vector<bool> visited;
};

int main() {
    LiverModelDiscrete model;
    int num_tcells = 40693;  // num of cells
    
    model.initialize_tcells(num_tcells);
    model.burn_in();
    
    auto result = model.simulate_coverage(40, 1000);  // Fixed: C++11 style decomposition
    std::vector<float> coverage = result.first;
    std::vector<float> times = result.second;
    
    float final_time_hours = times.back() / 60.0f;
    std::cout << "Time to reach 40% coverage: " << final_time_hours*10*3.83 << " hours\n with " << num_tcells << " T-cells\n";
    
    model.plot_results(coverage, times);
    return 0;
}
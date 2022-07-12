#include <pybind11/pybind11.h>

#include "iostream"
#include "cassert"
#include "omp.h"
#include "set"
#include "unordered_map"
#include "vector"
#include "pybind11/stl.h"
#include "cmath"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include "cassert"
//#include "eigen-3.4.0/Eigen/Dense"
//#include "eigen-3.4.0/Eigen/Eigenvalues"



const int32_t inf = 1000000000; //1e9

using Image2d = std::vector<std::vector<double>>;
using Image3d = std::vector<std::vector<std::vector<double>>>;

bool check_size_2d(Image2d &image) {
    if (image.empty()) {
        return false;
    }
    if (image[0].empty()) {
        return false;
    }
#pragma omp parallel for num_threads(4)
    for (size_t i = 1; i < image.size(); ++i) {
        if (image[i].size() != image[0].size()) {
            return false;
        }
    }
    return true;
}

bool check_size_3d(Image3d &image) {
    if (image.empty()) {
        return false;
    }
    if (image[0].empty()) {
        return false;
    }
#pragma omp parallel for num_threads(4)
    for (size_t i = 1; i < image.size(); ++i) {
        if (image[i].size() != image[0].size()) {
            return false;
        }
    }
#pragma omp parallel for num_threads(4) collapse(2)
    for (size_t i = 0; i < image.size(); ++i) {
        for (int j = 0; j < image[i].size(); ++j) {
            if (image[i][j].size() != image[0][0].size()) {
                return false;
            }
        }
    }
    return true;
}

//bool check_transformation_matrix_2d(std::vector<std::vector<double>> &transformation) {
//    if (transformation.size() != 2) {
//        return false;
//    }
//    if (transformation[0].size() != 2 || transformation[1].size() != 2) {
//        return false;
//    }
//    Eigen::Matrix<double, 2, 2> transformation_matrix{{transformation[0][0], transformation[0][1]},
//                                                      {transformation[1][0], transformation[1][1]}};
//    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> solver(transformation_matrix);
//    return solver.eigenvalues()[0].real() > 0 && solver.eigenvalues()[1].real() > 0;
//}
//
//bool check_transformation_matrix_3d(std::vector<std::vector<double>> &transformation) {
//    if (transformation.size() != 3) {
//        return false;
//    }
//    if (transformation[0].size() != 3 || transformation[1].size() != 3 || transformation[2].size() != 3) {
//        return false;
//    }
//    Eigen::Matrix<double, 3, 3> transformation_matrix{{transformation[0][0], transformation[0][1], transformation[0][2]},
//                                                      {transformation[1][0], transformation[1][1], transformation[1][2]},
//                                                      {transformation[2][0], transformation[2][1], transformation[2][2]}};
//    Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> solver(transformation_matrix);
//    return solver.eigenvalues()[0].real() > 0 && solver.eigenvalues()[1].real() > 0 &&
//           solver.eigenvalues()[2].real() > 0;
//}

int32_t get_vertex_id_2d(Image2d &image, int32_t x, int32_t y) {
    return x * static_cast<int32_t>(image[0].size()) + y;
}

int32_t get_vertex_id_3d(Image3d &image, int32_t x, int32_t y, int32_t z) {
    return x * static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()) +
           y * static_cast<int32_t>(image[0][0].size()) + z;
}

int32_t get_x_in_image_2d(Image2d &image, int32_t vertex_id) {
    return vertex_id / static_cast<int32_t>(image[0].size());
}

int32_t get_y_in_image_2d(Image2d &image, int32_t vertex_id) {
    return vertex_id % static_cast<int32_t>(image[0].size());
}

int32_t get_x_in_image_3d(Image3d &image, int32_t vertex_id) {
    return vertex_id / (static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()));
}

int32_t get_y_in_image_3d(Image3d &image, int32_t vertex_id) {
    return (vertex_id % (static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()))) /
           static_cast<int32_t>(image[0][0].size());
}

int32_t get_z_in_image_3d(Image3d &image, int32_t vertex_id) {
    return (vertex_id % (static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()))) %
           static_cast<int32_t>(image[0][0].size());
}

double calculate_distance_2d(std::pair<int32_t, int32_t> v1, std::pair<int32_t, int32_t> v2,
                             std::vector<std::vector<double>> &transformation) {
    v1.first -= v2.first;
    v1.second -= v2.second;
    return sqrt(1.0 * (v1.first * transformation[0][0] + v1.second * transformation[1][0]) * v1.first +
                (v1.first * transformation[0][1] + v1.second * transformation[1][1]) * v1.second);
}

double calculate_distance_3d(std::vector<double> &v1, std::vector<double> &v2,
                             std::vector<std::vector<double>> &transformation) {
    for (int32_t i = 0; i < 3; ++i) {
        v1[i] -= v2[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        v2[i] = v1[0] * transformation[0][i] + v1[1] * transformation[1][i] + v1[2] * transformation[2][i];
    }
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void
get_neighbours_4_connectivity_2d(Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, std::vector<std::vector<double>> &transformation) {
    int32_t int_x = get_x_in_image_2d(image, vertex);
    int32_t int_y = get_y_in_image_2d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i + j) == 1) {
                int32_t new_x = int_x + i;
                int32_t new_y = int_y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    neighbours.emplace_back(get_vertex_id_2d(image, new_x, new_y),
                                            calculate_distance_2d(std::make_pair(int_x, int_y),
                                                                  std::make_pair(new_x, new_y), transformation));
                }
            }
        }
    }
}

void
get_neighbours_6_connectivity_3d(Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, std::vector<std::vector<double>> &transformation) {
    int32_t int_x = get_x_in_image_3d(image, vertex);
    int32_t int_y = get_y_in_image_3d(image, vertex);
    int32_t int_z = get_z_in_image_3d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(3)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            for (int32_t k = 0; k <= 1; ++k) {
                if (abs(i) + abs(j) + abs(k) == 1) {
                    int32_t new_x = int_x + i;
                    int32_t new_y = int_y + j;
                    int32_t new_z = int_z + k;
                    if (new_x >= 0 && new_y >= 0 && new_z >= 0 && new_x < image.size() && new_y < image[0].size() &&
                        new_z < image[0][0].size()) {
                        std::vector<double> v1 = {static_cast<double >(int_x), static_cast<double >(int_y),
                                                  static_cast<double >(int_z)};
                        std::vector<double> v2 = {static_cast<double >(new_x), static_cast<double >(new_y),
                                                  static_cast<double >(new_z)};
                        neighbours.emplace_back(get_vertex_id_3d(image, new_x, new_y, new_z),
                                                calculate_distance_3d(v1, v2, transformation));
                    }
                }
            }
        }
    }
}

void get_neighbours_diagonal_connectivity_2d(Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                             int32_t vertex, std::vector<std::vector<double>> &transformation) {
    int32_t int_x = get_x_in_image_2d(image, vertex);
    int32_t int_y = get_y_in_image_2d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i) == 1 && abs(j) == 1) {
                int32_t new_x = int_x + i;
                int32_t new_y = int_y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    neighbours.emplace_back(get_vertex_id_2d(image, new_x, new_y),
                                            calculate_distance_2d(std::make_pair(int_x, int_y),
                                                                  std::make_pair(new_x, new_y), transformation));
                }
            }
        }
    }
}

void get_neighbours_diagonal_connectivity_3d(Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                             int32_t vertex, std::vector<std::vector<double>> &transformation) {
    int32_t int_x = get_x_in_image_3d(image, vertex);
    int32_t int_y = get_y_in_image_3d(image, vertex);
    int32_t int_z = get_z_in_image_3d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(3)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            for (int32_t k = 0; k <= 1; ++k) {
                if (abs(i) + abs(j) + abs(k) >= 2) {
                    int32_t new_x = int_x + i;
                    int32_t new_y = int_y + j;
                    int32_t new_z = int_z + k;
                    if (new_x >= 0 && new_y >= 0 && new_z >= 0 && new_x < image.size() && new_y < image[0].size() &&
                        new_z < image[0][0].size()) {
                        std::vector<double> v1 = {static_cast<double >(int_x), static_cast<double >(int_y),
                                                  static_cast<double >(int_z)};
                        std::vector<double> v2 = {static_cast<double >(new_x), static_cast<double >(new_y),
                                                  static_cast<double >(new_z)};
                        neighbours.emplace_back(get_vertex_id_3d(image, new_x, new_y, new_z),
                                                calculate_distance_3d(v1, v2, transformation));
                    }
                }
            }
        }
    }
}

void
get_neighbours_8_connectivity_2d(Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, std::vector<std::vector<double>> &transformation) {
    get_neighbours_4_connectivity_2d(image, neighbours, vertex, transformation);
    get_neighbours_diagonal_connectivity_2d(image, neighbours, vertex, transformation);
}

void
get_neighbours_26_connectivity_3d(Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                  int32_t vertex, std::vector<std::vector<double>> &transformation) {
    get_neighbours_6_connectivity_3d(image, neighbours, vertex, transformation);
    get_neighbours_diagonal_connectivity_3d(image, neighbours, vertex, transformation);
}


bool
is_border_2d(Image2d &image, std::vector<std::vector<std::pair<int32_t, double>>> &image_graph, int32_t vertex,
             bool black) {
#pragma omp parallel for num_threads(4)
    for (auto i: image_graph[vertex]) {
        int32_t x = get_x_in_image_2d(image, i.first);
        int32_t y = get_y_in_image_2d(image, i.first);
        if (image[x][y] == !black) {
            return true;
        }
    }
    return false;
}

bool
is_border_3d(Image3d &image, std::vector<std::vector<std::pair<int32_t, double>>> &image_graph, int32_t vertex,
             bool black) {
#pragma omp parallel for num_threads(4)
    for (auto i: image_graph[vertex]) {
        int32_t x = get_x_in_image_3d(image, i.first);
        int32_t y = get_y_in_image_3d(image, i.first);
        int32_t z = get_z_in_image_3d(image, i.first);
        if (image[x][y][z] == !black) {
            return true;
        }
    }
    return false;
}

void build_graph_2d(Image2d &image,
                    std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                    std::vector<std::vector<double>> &transformation, std::string &connectivity_type) {
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        if (connectivity_type == "4-connectivity") {
            get_neighbours_4_connectivity_2d(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "8-connectivity") {
            get_neighbours_8_connectivity_2d(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "diagonal-connectivity") {
            get_neighbours_diagonal_connectivity_2d(image, image_graph[i], i, transformation);
        } else {
            assert(0);
        }
    }
}

void build_graph_3d(Image3d &image,
                    std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                    std::vector<std::vector<double>> &transformation, std::string &connectivity_type) {
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        if (connectivity_type == "6-connectivity") {
            get_neighbours_6_connectivity_3d(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "26-connectivity") {
            get_neighbours_26_connectivity_3d(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "diagonal-connectivity") {
            get_neighbours_diagonal_connectivity_3d(image, image_graph[i], i, transformation);
        } else {
            assert(0);
        }
    }
}

void update_distances(std::vector<int32_t> &border,
                      std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                      std::vector<double> &distances) {

    int32_t bucket_size = static_cast<int32_t>(image_graph.size()) / 10 + 1;
    int32_t left_bucket = 0;

    std::vector<std::set<std::pair<double, int32_t>>> buckets(image_graph.size() / bucket_size + 1);

    std::set<std::pair<int32_t, int32_t>> priority_queue;
#pragma omp parallel for num_threads(4)
    for (auto &i: border) {
        buckets[0].insert({0, i});
        distances[i] = 0;
    }

    while (left_bucket < buckets.size()) {
        double current_distance = buckets[left_bucket].begin()->first;
        int32_t current_vertex = buckets[left_bucket].begin()->second;
        buckets[left_bucket].erase(buckets[left_bucket].begin());
#pragma omp parallel for num_threads(4)
        for (auto go_to: image_graph[current_vertex]) {
            //int32_t current_x = get_x_in_image(transformed_image, go_to.first);
            //int32_t current_y = get_y_in_image(transformed_image, go_to.first);
            if (distances[go_to.first] > current_distance + go_to.second) {
                int32_t new_bucket_id = std::floor((current_distance + go_to.second) / bucket_size);
                int32_t previous_bucket_id;
                if (distances[go_to.first] == inf) {
                    previous_bucket_id = static_cast<int32_t>(buckets.size()) - 1;
                } else {
                    previous_bucket_id = std::floor(distances[go_to.first] / bucket_size);
                }
                buckets[previous_bucket_id].erase({distances[go_to.first], go_to.first});
                buckets[new_bucket_id].insert({current_distance + go_to.second, go_to.first});
                distances[go_to.first] = current_distance + go_to.second;
            }
        }
#pragma omp parallel for num_threads(4)
        for (int32_t i = left_bucket; i < buckets.size(); ++i) {
            if (!buckets[i].empty()) {
                break;
            }
            left_bucket++;
        }
    }
}

Image2d make_transformation_2d(Image2d &image,
                               Image2d transformation = {{1.0, 0.0},
                                                         {0.0, 1.0}},
                               std::string connectivity_type = "8-connectivity",
                               bool is_signed = false) {
    assert(check_size_2d(image));
    assert(check_transformation_matrix_2d(transformation));

    Image2d transformed_image(image.size());
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image.size(); ++i) {
        transformed_image[i].assign(image[i].size(), inf);
    }

    std::vector<std::vector<std::pair<int32_t, double>>> image_graph;
    image_graph.resize(image[0].size() * image.size());
    build_graph_2d(image, image_graph, transformation, connectivity_type);

    std::vector<int32_t> border;
    std::vector<double> distances(image_graph.size(), inf);
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        if (image[current_x][current_y] != 0) {
            transformed_image[current_x][current_y] = 0;
            distances[i] = 0;
            if (is_border_2d(image, image_graph, i, true)) {
                border.push_back(i);
            }
        }
    }
    update_distances(border, image_graph, distances);
    for (int32_t i = 0; i < distances.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        transformed_image[current_x][current_y] = distances[i];
    }
    if (is_signed) {
        border.clear();
        distances.assign(distances.size(), inf);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image_graph.size(); ++i) {
            int32_t current_x = get_x_in_image_2d(image, i);
            int32_t current_y = get_y_in_image_2d(image, i);
            if (image[current_x][current_y] == 0) {
                distances[i] = 0;
                if (is_border_2d(image, image_graph, i, false)) {
                    border.push_back(i);
                }
            }
        }
        update_distances(border, image_graph, distances);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < distances.size(); ++i) {
            int32_t current_x = get_x_in_image_2d(image, i);
            int32_t current_y = get_y_in_image_2d(image, i);
            transformed_image[current_x][current_y] -= distances[i];
        }
    }
    return transformed_image;
}
Image3d make_transformation_3d(Image3d &image,
                               Image2d transformation = {{1.0, 0.0, 0.0},
                                                         {0.0, 1.0, 0.0},
                                                         {0.0, 0.0, 1.0}},
                               std::string connectivity_type = "6-connectivity",
                               bool is_signed = false) {
    assert(check_size_3d(image));
    assert(check_transformation_matrix_3d(transformation));

    Image3d transformed_image(image.size());
#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = 0; i < image.size(); ++i) {
        for (int32_t j = 0; j < image.size(); ++j) {
            transformed_image[i][j].assign(image[i][j].size(), inf);
        }
    }

    std::vector<std::vector<std::pair<int32_t, double>>> image_graph;
    image_graph.resize(image[0].size() * image.size() * image[0][0].size());
    build_graph_3d(image, image_graph, transformation, connectivity_type);

    std::vector<int32_t> border;
    std::vector<double> distances(image_graph.size(), inf);
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image_3d(image, i);
        int32_t current_y = get_y_in_image_3d(image, i);
        int32_t current_z = get_z_in_image_3d(image, i);
        if (image[current_x][current_y][current_z] != 0) {
            transformed_image[current_x][current_y][current_z] = 0;
            distances[i] = 0;
            if (is_border_3d(image, image_graph, i, true)) {
                border.push_back(i);
            }
        }
    }
    update_distances(border, image_graph, distances);
    for (int32_t i = 0; i < distances.size(); ++i) {
        int32_t current_x = get_x_in_image_3d(image, i);
        int32_t current_y = get_y_in_image_3d(image, i);
        int32_t current_z = get_z_in_image_3d(image, i);
        transformed_image[current_x][current_y][current_z] = distances[i];
    }
    if (is_signed) {
        border.clear();
        distances.assign(distances.size(), inf);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image_graph.size(); ++i) {
            int32_t current_x = get_x_in_image_3d(image, i);
            int32_t current_y = get_y_in_image_3d(image, i);
            int32_t current_z = get_z_in_image_3d(image, i);
            if (image[current_x][current_y][current_z] == 0) {
                distances[i] = 0;
                if (is_border_3d(image, image_graph, i, false)) {
                    border.push_back(i);
                }
            }
        }
        update_distances(border, image_graph, distances);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < distances.size(); ++i) {
            int32_t current_x = get_x_in_image_3d(image, i);
            int32_t current_y = get_y_in_image_3d(image, i);
            int32_t current_z = get_z_in_image_3d(image, i);
            transformed_image[current_x][current_y][current_z] -= distances[i];
        }
    }
    return transformed_image;
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");


    m.def("make_transformation_2d", &make_transformation_2d, R"pbdoc(
    )pbdoc");

    m.def("make_transformation_3d", &make_transformation_3d, R"pbdoc(
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
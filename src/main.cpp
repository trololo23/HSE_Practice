#include <pybind11/pybind11.h>

#include "iostream"
#include "cassert"
#include "omp.h"
#include "set"
#include "unordered_map"
#include "vector"
#include "pybind11/stl.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include "cassert"
//#include "eigen-3.4.0/Eigen/Dense"
//#include "eigen-3.4.0/Eigen/Eigenvalues"
#include "omp.h"
#include "cmath"
#include "queue"
#include "set"
#include "unordered_map"
#include "vector"

const int32_t inf = 1000000000; //1e9

using Image = std::vector<std::vector<double>>;

bool check_size(Image &image) {
    if (image.empty()) {
        return false;
    }
    for (size_t i = 1; i < image.size(); ++i) {
        if (image[i].size() != image[0].size()) {
            return false;
        }
    }
    return true;
}

//bool check_transformation_matrix(std::vector<std::vector<double>> &transformation) {
//    if (transformation.size() != 2) {
//        return false;
//    }
//    Eigen::Matrix<double, 2, 2> transformation_matrix{{transformation[0][0], transformation[0][1]}, {transformation[1][0], transformation[1][1]}};
//    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> solver(transformation_matrix);
//    return solver.eigenvalues()[0].real() > 0 && solver.eigenvalues()[1].real() > 0;
//}

int32_t get_vertex_id(Image &image, int32_t x, int32_t y) {
    return x * static_cast<int32_t>(image[0].size()) + y;
}

int32_t get_x_in_image(Image &image, int32_t vertex_id) {
    return vertex_id / static_cast<int32_t>(image[0].size());
}

int32_t get_y_in_image(Image &image, int32_t vertex_id) {
    return vertex_id % static_cast<int32_t>(image[0].size());
}

double calculate_distance(std::pair<int32_t, int32_t> v1, std::pair<int32_t, int32_t> v2,
                          std::vector<std::vector<double>> &transformation) {
    v1.first -= v2.first;
    v1.second -= v2.second;
    return sqrt(1.0 * (v1.first * transformation[0][0] + v1.second * transformation[1][0]) * v1.first +
                (v1.first * transformation[0][1] + v1.second * transformation[1][1]) * v1.second);
}

void
get_neighbours_4_connectivity(Image &image, std::vector<std::pair<int32_t, double>> &neighbours,
                              int32_t vertex, std::vector<std::vector<double>> &transformation) {
    int32_t int_x = get_x_in_image(image, vertex);
    int32_t int_y = get_y_in_image(image, vertex);

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i + j) == 1) {
                int32_t new_x = int_x + i;
                int32_t new_y = int_y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    neighbours.emplace_back(get_vertex_id(image, new_x, new_y),
                                            calculate_distance(std::make_pair(int_x, int_y),
                                                               std::make_pair(new_x, new_y), transformation));
                }
            }
        }
    }
}

void get_neighbours_diagonal_connectivity(Image &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                          int32_t vertex, std::vector<std::vector<double>> &transformation) {
    int32_t int_x = get_x_in_image(image, vertex);
    int32_t int_y = get_y_in_image(image, vertex);

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i) == 1 && abs(j) == 1) {
                int32_t new_x = int_x + i;
                int32_t new_y = int_y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    neighbours.emplace_back(get_vertex_id(image, new_x, new_y),
                                            calculate_distance(std::make_pair(int_x, int_y),
                                                               std::make_pair(new_x, new_y), transformation));
                }
            }
        }
    }
}

void
get_neighbours_8_connectivity(Image &image, std::vector<std::pair<int32_t, double>> &neighbours,
                              int32_t vertex, std::vector<std::vector<double>> &transformation) {
    get_neighbours_4_connectivity(image, neighbours, vertex, transformation);
    get_neighbours_diagonal_connectivity(image, neighbours, vertex, transformation);
}


bool
is_border(Image &image, std::vector<std::vector<std::pair<int32_t, double>>> &image_graph, int32_t vertex, bool black) {
#pragma omp parallel for num_threads(4)
    for (auto i: image_graph[vertex]) {
        int32_t x = get_x_in_image(image, i.first);
        int32_t y = get_y_in_image(image, i.first);
        if (image[x][y] == !black) {
            return true;
        }
    }
    return false;
}

void build_graph(Image &image,
                 std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                 Image &transformed_image,
                 std::vector<std::vector<double>> &transformation, std::string &connectivity_type) {
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        if (connectivity_type == "4-connectivity") {
            get_neighbours_4_connectivity(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "8-connectivity") {
            get_neighbours_8_connectivity(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "diagonal-connectivity") {
            get_neighbours_diagonal_connectivity(image, image_graph[i], i, transformation);
        } else {
            assert(0);
        }
    }

#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image(image, i);
        int32_t current_y = get_y_in_image(image, i);
        if (image[current_x][current_y] > 0) {
            transformed_image[current_x][current_y] = inf;
        }
    }
}

void update_distances(std::vector<int32_t> &border, std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                      std::vector<std::vector<double>> &transformed_image) {

    int32_t bucket_size = static_cast<int32_t>(image_graph.size()) / 10 + 1;
    int32_t left_bucket = 0;

    std::vector<std::set<std::pair<double, int32_t>>> buckets(image_graph.size() / bucket_size + 1);

    std::set<std::pair<int32_t, int32_t>> priority_queue;
#pragma omp parallel for num_threads(4)
    for (auto &i: border) {
        buckets[0].insert({0, i});
    }

    while (left_bucket < buckets.size()) {
        double current_distance = buckets[left_bucket].begin()->first;
        int32_t current_vertex = buckets[left_bucket].begin()->second;
        buckets[left_bucket].erase(buckets[left_bucket].begin());
#pragma omp parallel for num_threads(4)
        for (auto go_to: image_graph[current_vertex]) {
            int32_t current_x = get_x_in_image(transformed_image, go_to.first);
            int32_t current_y = get_y_in_image(transformed_image, go_to.first);
            if (transformed_image[current_x][current_y] > current_distance + go_to.second) {
                int32_t new_bucket_id = std::floor((current_distance + go_to.second) / bucket_size);
                int32_t previous_bucket_id;
                if (transformed_image[current_x][current_y] == inf) {
                    previous_bucket_id = static_cast<int32_t>(buckets.size()) - 1;
                } else {
                    previous_bucket_id = std::floor((transformed_image[current_x][current_y]) / bucket_size);
                }
                buckets[previous_bucket_id].erase({transformed_image[current_x][current_y], go_to.first});
                buckets[new_bucket_id].insert({current_distance + go_to.second, go_to.first});
                transformed_image[current_x][current_y] = current_distance + go_to.second;
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

Image make_transformation(Image &image,
                          Image transformation = {{1.0, 0.0},
                                                  {0.0, 1.0}},
                          std::string connectivity_type = "8-connectivity",
                          bool is_signed = false) {
    assert(check_size(image));
    assert(check_transformation_matrix(transformation));

    Image transformed_image(image.size());
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image.size(); ++i) {
        transformed_image[i].resize(image[i].size());
    }

    std::vector<std::vector<std::pair<int32_t, double>>> image_graph;
    image_graph.resize(image[0].size() * image.size());
    build_graph(image, image_graph, transformed_image, transformation, connectivity_type);

    std::vector<int32_t> border;
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image(image, i);
        int32_t current_y = get_y_in_image(image, i);
        if (image[current_x][current_y] != 0) {
            transformed_image[current_x][current_y] = 0;
            if (is_border(image, image_graph, i, true)) {
                border.push_back(i);
            }
        }
    }
    update_distances(border, image_graph, transformed_image);
    if (is_signed) {
        border.clear();
        Image transformed_image_signed(image.size());
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image.size(); ++i) {
            transformed_image_signed[i].resize(image[i].size());
        }

#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image_graph.size(); ++i) {
            int32_t current_x = get_x_in_image(image, i);
            int32_t current_y = get_y_in_image(image, i);
            if (image[current_x][current_y] != 0) {
                transformed_image_signed[current_x][current_y] = inf;
            }
        }
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image_graph.size(); ++i) {
            int32_t current_x = get_x_in_image(image, i);
            int32_t current_y = get_y_in_image(image, i);
            if (image[current_x][current_y] == 0) {
                transformed_image_signed[current_x][current_y] = 0;
                if (is_border(image, image_graph, i, false)) {
                    border.push_back(i);
                }
            }
        }
        update_distances(border, image_graph, transformed_image_signed);
#pragma omp parallel for num_threads(4) collapse(2)
        for (int32_t i = 0; i < transformed_image_signed.size(); ++i) {
            for (int32_t j = 0; j < transformed_image_signed[i].size(); ++j) {
                transformed_image[i][j] -= transformed_image_signed[i][j];
            }
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

    m.def("check_size", &check_size, R"pbdoc(
    )pbdoc");

    m.def("get_vertex_id", &get_vertex_id, R"pbdoc(
    )pbdoc");

    m.def("get_x_in_image", &get_x_in_image, R"pbdoc(
    )pbdoc");

    m.def("get_y_in_image", &get_y_in_image, R"pbdoc(
    )pbdoc");

    m.def("get_neighbours_4_connectivity", &get_neighbours_4_connectivity, R"pbdoc(
    )pbdoc");

    m.def("get_neighbours_diagonal_connectivity", &get_neighbours_diagonal_connectivity, R"pbdoc(
    )pbdoc");

    m.def("get_neighbours_8_connectivity", &get_neighbours_8_connectivity, R"pbdoc(
    )pbdoc");

    m.def("is_border", &is_border, R"pbdoc(
    )pbdoc");

    m.def("build_graph", &build_graph, R"pbdoc(
    )pbdoc");

    m.def("update_distances", &update_distances, R"pbdoc(
    )pbdoc");

//    m.def("euclidean_transformation", &euclidean_transformation, R"pbdoc(
//    )pbdoc");

//    m.def("signed_euclidean_distance", &get_vertex_id, R"pbdoc(
//    )pbdoc");

    m.def("make_transformation", &make_transformation, R"pbdoc(
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
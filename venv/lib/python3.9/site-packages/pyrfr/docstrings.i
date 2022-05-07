
// File: index.xml

// File: classrfr_1_1data__containers_1_1array__data__container.xml


%feature("docstring") rfr::data_containers::array_data_container "
";

%feature("docstring") rfr::data_containers::array_data_container::array_data_container "
`array_data_container(num_type *features, response_type *responses, index_type
    *types, index_type n_data_points, index_type n_features)`  
";

%feature("docstring") rfr::data_containers::array_data_container::~array_data_container "
`~array_data_container()`  
";

%feature("docstring") rfr::data_containers::array_data_container::feature "
`feature(index_type feature_index, index_type sample_index) const -> num_type`  
";

%feature("docstring") rfr::data_containers::array_data_container::features "
`features(index_type feature_index, std::vector< index_type > &sample_indices)
    const -> std::vector< num_type >`  
";

%feature("docstring") rfr::data_containers::array_data_container::response "
`response(index_type sample_index) const -> response_type`  
";

%feature("docstring") rfr::data_containers::array_data_container::add_data_point "
`add_data_point(num_type *features, index_type num_elements, response_type
    response)`  
";

%feature("docstring") rfr::data_containers::array_data_container::retrieve_data_point "
`retrieve_data_point(index_type index) const -> std::vector< num_type >`  
";

%feature("docstring") rfr::data_containers::array_data_container::get_type_of_feature "
`get_type_of_feature(index_type feature_index) const -> index_type`  
";

%feature("docstring") rfr::data_containers::array_data_container::set_type_of_feature "
`set_type_of_feature(index_type feature_index, index_type feature_type)`  
";

%feature("docstring") rfr::data_containers::array_data_container::get_type_of_response "
`get_type_of_response() const -> index_type`  
";

%feature("docstring") rfr::data_containers::array_data_container::set_type_of_response "
`set_type_of_response(index_type resp_t)`  
";

%feature("docstring") rfr::data_containers::array_data_container::num_features "
`num_features() const -> index_type`  
";

%feature("docstring") rfr::data_containers::array_data_container::num_data_points "
`num_data_points() const -> index_type`  
";

// File: classrfr_1_1data__containers_1_1base.xml


%feature("docstring") rfr::data_containers::base "

The interface for any data container with the minimal functionality.  

C++ includes: data_container.hpp
";

%feature("docstring") rfr::data_containers::base::~base "
`~base()`  
";

%feature("docstring") rfr::data_containers::base::feature "
`feature(index_t feature_index, index_t sample_index) const =0 -> num_t`  

Function for accessing a single feature value, consistency checks might be
omitted for performance.  

Parameters
----------
* `feature_index` :  
    The index of the feature requested  
* `sample_index` :  
    The index of the data point.  

Returns
-------
the stored value  
";

%feature("docstring") rfr::data_containers::base::features "
`features(index_t feature_index, const std::vector< index_t > &sample_indices)
    const =0 -> std::vector< num_t >`  

member function for accessing the feature values of multiple data points at
once, consistency checks might be omitted for performance  

Parameters
----------
* `feature_index` :  
    The index of the feature requested  
* `sample_indices` :  
    The indices of the data point.  

Returns
-------
the stored values  
";

%feature("docstring") rfr::data_containers::base::response "
`response(index_t sample_index) const =0 -> response_t`  

member function to query a single response value, consistency checks might be
omitted for performance  

Parameters
----------
* `sample_index` :  
    the response of which data point  

Returns
-------
the response value  
";

%feature("docstring") rfr::data_containers::base::predict_value "
`predict_value(index_t sample_index) const =0 -> response_t`  

member function to query a single response value, consistency checks might be
omitted for performance  

Parameters
----------
* `sample_index` :  
    the response of which data point  

Returns
-------
the prediction value  
";

%feature("docstring") rfr::data_containers::base::weight "
`weight(index_t sample_index) const =0 -> num_t`  

function to access the weight attributed to a single data point  

Parameters
----------
* `sample_index` :  
    which data point  

Returns
-------
the weigth of that sample  
";

%feature("docstring") rfr::data_containers::base::add_data_point "
`add_data_point(std::vector< num_t > features, response_t response, num_t
    weight)=0`  

method to add a single data point  

Parameters
----------
* `features` :  
    a vector containing the features  
* `response` :  
    the corresponding response value  
* `weight` :  
    the weight of the data point  
";

%feature("docstring") rfr::data_containers::base::add_data_point "
`add_data_point(std::vector< num_t > features, std::vector< response_t >
    response, num_t weight)=0`  

method to add a single data point  

Parameters
----------
* `features` :  
    a vector containing the features  
* `response` :  
    the corresponding response vector. The first entry is the value used to fit
    the trees with, the second is used to predict.  
* `weight` :  
    the weight of the data point  
";

%feature("docstring") rfr::data_containers::base::retrieve_data_point "
`retrieve_data_point(index_t index) const =0 -> std::vector< num_t >`  

method to retrieve a data point  

Parameters
----------
* `index` :  
    index of the datapoint to extract  

Returns
-------
std::vector<num_t> the features of the data point  
";

%feature("docstring") rfr::data_containers::base::get_type_of_feature "
`get_type_of_feature(index_t feature_index) const =0 -> index_t`  

query the type of a feature  

Parameters
----------
* `feature_index` :  
    the index of the feature  

Returns
-------
int type of the feature: 0 - numerical value (float or int); n>0 - categorical
value with n different values {0,1,...,n-1}  
";

%feature("docstring") rfr::data_containers::base::get_type_of_response "
`get_type_of_response() const =0 -> index_t`  

query the type of the response  

Returns
-------
index_t type of the response: 0 - numerical value (float or int); n>0 -
categorical value with n different values {0,1,...,n-1}  
";

%feature("docstring") rfr::data_containers::base::set_type_of_feature "
`set_type_of_feature(index_t feature_index, index_t feature_type)=0`  

specifying the type of a feature  

Parameters
----------
* `feature_index` :  
    the index of the feature whose type is specified  
* `feature_type` :  
    the actual type (0 - numerical, value >0 catergorical with values from
    {0,1,...value-1}  
";

%feature("docstring") rfr::data_containers::base::set_type_of_response "
`set_type_of_response(index_t response_type)=0`  

specifying the type of the response  

Parameters
----------
* `response_type` :  
    the actual type (0 - numerical, value >0 catergorical with values from
    {0,1,...value-1}  
";

%feature("docstring") rfr::data_containers::base::set_bounds_of_feature "
`set_bounds_of_feature(index_t feature_index, num_t min, num_t max)=0`  

specifies the interval of allowed values for a feature  

To marginalize out certain feature dimensions using non-i.i.d. data, the
numerical bounds on each variable have to be known. This only applies to
numerical features.  

Note: The forest will not check if a datapoint is consistent with the specified
bounds!  

Parameters
----------
* `feature_index` :  
    feature_index the index of the feature  
* `min` :  
    the smallest value for the feature  
* `max` :  
    the largest value for the feature  
";

%feature("docstring") rfr::data_containers::base::get_bounds_of_feature "
`get_bounds_of_feature(index_t feature_index) const =0 -> std::pair< num_t,
    num_t >`  

query the allowed interval for a feature; applies only to continuous variables  

Parameters
----------
* `feature_index` :  
    the index of the feature  

Returns
-------
std::pair<num_t,num_t> interval of allowed values  
";

%feature("docstring") rfr::data_containers::base::num_features "
`num_features() const =0 -> index_t`  

the number of features of every datapoint in the container  
";

%feature("docstring") rfr::data_containers::base::num_data_points "
`num_data_points() const =0 -> index_t`  

the number of data points in the container  
";

// File: classrfr_1_1trees_1_1binary__f_a_n_o_v_a__tree.xml


%feature("docstring") rfr::trees::binary_fANOVA_tree "
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::binary_fANOVA_tree "
`binary_fANOVA_tree()`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::~binary_fANOVA_tree "
`~binary_fANOVA_tree()`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data,
    rfr::trees::tree_options< num_t, response_t, index_t > tree_opts, const
    std::vector< num_t > &sample_weights, rng_t &rng)`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::marginalized_prediction_stat "
`marginalized_prediction_stat(const std::vector< num_t > &feature_vector,
    std::vector< std::vector< num_t > > pcs, std::vector< index_t > types) const
    -> rfr::util::weighted_running_statistics< num_t >`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::precompute_marginals "
`precompute_marginals(num_t l_cutoff, num_t u_cutoff, const std::vector<
    std::vector< num_t > > &pcs, const std::vector< index_t > &types)`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::all_split_values "
`all_split_values(const std::vector< index_t > &types) -> std::vector<
    std::vector< num_t > >`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_mean "
`get_mean() const -> num_t`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_total_variance "
`get_total_variance() const -> num_t`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_subspace_size "
`get_subspace_size(index_t node_index) const -> num_t`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_marginal_prediction "
`get_marginal_prediction(index_t node_index) const -> num_t`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_marginal_prediction_stat "
`get_marginal_prediction_stat(index_t node_index) const ->
    rfr::util::weighted_running_statistics< num_t >`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_active_variables "
`get_active_variables(index_t node_index) const -> const std::vector< bool > &`  
";

%feature("docstring") rfr::trees::binary_fANOVA_tree::get_nodes "
`get_nodes() const -> const std::vector< rfr::nodes::k_ary_node_full< 2,
    split_t, num_t, response_t, index_t, rng_t > > &`  
";

// File: classrfr_1_1splits_1_1binary__split__one__feature__gini.xml


%feature("docstring") rfr::splits::binary_split_one_feature_gini "

OUTDATED: Needs to be adapted to the new internal API for splitting!  

C++ includes: classification_split.hpp
";

%feature("docstring") rfr::splits::binary_split_one_feature_gini::find_best_split "
`find_best_split(const rfr::data_containers::data_container_base< num_t,
    response_t, index_t > &data, const std::vector< index_t > &features_to_try,
    std::vector< index_t > &indices, std::array< typename std::vector< index_t
    >::iterator, 3 > &split_indices_it, rng_t &rng) -> num_t`  

the implementation to find the best binary split using only one feature
minimizing the RSS loss  

The best binary split is determined among all allowed features. For a continuous
feature the split is a single value. For catergoricals, the split criterion is a
\"set\" (actual implementation might use a different datatype for performance).
In both cases the split is computed as efficiently as possible exploiting
properties of the RSS loss (optimal split for categoricals can be found in
polynomial rather than exponential time in the number of possible values). The
constructor assumes that the data should be split. Testing whether the number of
points and their values allow further splitting is checked by the tree  

Parameters
----------
* `data` :  
    the container holding the training data  
* `features_to_try` :  
    a vector with the indices of all the features that can be considered for
    this split  
* `indices` :  
    a vector containing the subset of data point indices to be considered
    (output!)  
* `an` :  
    iterator into this vector that says where to split the data for the two
    children  
* `rng` :  
    a pseudo random number generator instance  

> uses C++11 range based loop  

> uses C++11 lambda function, how exciting :)  
";

%feature("docstring") rfr::splits::binary_split_one_feature_gini::best_split_continuous "
`best_split_continuous(const rfr::data_containers::data_container_base< num_t,
    response_t, index_t > &data, const index_t &fi, std::vector< num_t >
    &split_criterion_copy, std::vector< index_t > &indices_copy, typename
    std::vector< index_t >::iterator &split_indices_it_copy) -> num_t`  

member function to find the best possible split for a single (continuous)
feature  

Parameters
----------
* `data` :  
    pointer to the the data container  
* `fi` :  
    the index of the feature to be used  
* `split_criterion_copy` :  
    a reference to store the split criterion  
* `indices_copy` :  
    a const reference to the indices (const b/c it has already been sorted)  
* `split_indices_it_copy` :  
    an iterator that will point to the first element of indices_copy that would
    go into the right child  

Returns
-------
the gini criterion of this split  
";

%feature("docstring") rfr::splits::binary_split_one_feature_gini::best_split_categorical "
`best_split_categorical(const rfr::data_containers::data_container_base< num_t,
    response_t, index_t > &data, const index_t &fi, const index_t
    &num_categories, std::vector< num_t > &split_criterion_copy, std::vector<
    index_t > &indices_copy, typename std::vector< index_t >::iterator
    &split_indices_it_copy, rng_t &rng) -> num_t`  

member function to find the best possible split for a single (categorical)
feature  

Parameters
----------
* `data` :  
    pointer to the the data container  
* `fi` :  
    the index of the feature to be used  
* `num_categories` :  
    how many different values this variable can take  
* `split_criterion_copy` :  
    a reference to store the split criterion  
* `indices_copy` :  
    a const reference to the indices (const b/c it has already been sorted)  
* `split_indices_it_copy` :  
    an iterator that will point to the first element of indices_copy that would
    go into the right child  

Returns
-------
float the loss of this split  

>assumes that the features for categoricals have been properly rounded so
casting them to ints results in the right value!  
";

%feature("docstring") rfr::splits::binary_split_one_feature_gini::print_info "
`print_info()`  

some debug output that prints a informative representation to std::cout  
";

%feature("docstring") rfr::splits::binary_split_one_feature_gini::latex_representation "
`latex_representation() -> std::string`  

member function to create a string representing the split criterion  

Returns
-------
std::string a label that characterizes the split  
";

%feature("docstring") rfr::splits::binary_split_one_feature_gini::get_split_criterion "
`get_split_criterion() -> std::vector< num_t >`  
";

// File: classrfr_1_1splits_1_1binary__split__one__feature__rss__loss.xml


%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss "
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::get_num_categories "
`get_num_categories() const -> index_t`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::find_best_split "
`find_best_split(const rfr::data_containers::base< num_t, response_t, index_t >
    &data, const std::vector< index_t > &features_to_try, typename std::vector<
    rfr::splits::data_info_t< num_t, response_t, index_t >>::iterator
    infos_begin, typename std::vector< rfr::splits::data_info_t< num_t,
    response_t, index_t >>::iterator infos_end, std::array< typename
    std::vector< rfr::splits::data_info_t< num_t, response_t, index_t
    >>::iterator, 3 > &info_split_its, index_t min_samples_in_child, num_t
    min_weight_in_child, rng_t &rng) -> num_t`  

the implementation to find the best binary split using only one feature
minimizing the RSS loss  

The best binary split is determined among all allowed features. For a continuous
feature the split is a single value. For catergoricals, the split criterion is a
\"set\" (actual implementation might use a different datatype for performance).
In both cases the split is computed as efficiently as possible exploiting
properties of the RSS loss (optimal split for categoricals can be found in
polynomial rather than exponential time in the number of possible values). The
constructor assumes that the data should be split. Testing whether the number of
points and their values allow further splitting is checked by the tree  

Parameters
----------
* `data` :  
    the container holding the training data  
* `features_to_try` :  
    a vector with the indices of all the features that can be considered for
    this split  
* `infos_begin` :  
    iterator to the first (relevant) element in a vector containing the minimal
    information in tuples  
* `infos_end` :  
    iterator beyond the last (relevant) element in a vector containing the
    minimal information in tuples  
* `info_split_its` :  
    iterators into this vector saying where to split the data for the two
    children  
* `min_samples_in_child` :  
    smallest acceptable number of samples in any of the children  
* `min_weight_in_child` :  
    smallest acceptable weight in any of the children  
* `rng` :  
    a random number generator instance  

Returns
-------
num_t loss of the best found split  

> uses C++11 range based loop  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::best_split_continuous "
`best_split_continuous(typename std::vector< rfr::splits::data_info_t< num_t,
    response_t, index_t >>::iterator infos_begin, typename std::vector<
    rfr::splits::data_info_t< num_t, response_t, index_t >>::iterator infos_end,
    num_t &split_value, rfr::util::weighted_running_statistics< num_t >
    right_stat, index_t min_samples_in_child, num_t min_weight_in_child, rng_t
    &rng) -> num_t`  

member function to find the best possible split for a single (continuous)
feature  

Parameters
----------
* `infos_begin` :  
    iterator to the first (relevant) element in a vector containing the minimal
    information in tuples  
* `infos_end` :  
    iterator beyond the last (relevant) element in a vector containing the
    minimal information in tuples  
* `split_value` :  
    a reference to store the split (numerical) criterion  
* `right_stat` :  
    a weighted_runnin_statistics object containing the statistics of all
    responses  
* `min_samples_in_child` :  
    smallest acceptable number of distinct data points in any of the children  
* `min_weight_in_child` :  
    smallest acceptable sum of all weights in any of the children  
* `rng` :  
    a pseudo random number generator instance  

Returns
-------
float the loss of this split  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::best_split_categorical "
`best_split_categorical(typename std::vector< rfr::splits::data_info_t< num_t,
    response_t, index_t >>::iterator infos_begin, typename std::vector<
    rfr::splits::data_info_t< num_t, response_t, index_t >>::iterator infos_end,
    index_t num_categories, std::bitset< max_num_categories > &split_set,
    rfr::util::weighted_running_statistics< num_t > right_stat, index_t
    min_samples_in_child, num_t min_weight_in_child, rng_t &rng) -> num_t`  

member function to find the best possible split for a single (categorical)
feature  

Parameters
----------
* `infos_begin` :  
    iterator to the first (relevant) element in a vector containing the minimal
    information in tuples  
* `infos_end` :  
    iterator beyond the last (relevant) element in a vector containing the
    minimal information in tuples *  
* `num_categories` :  
    the feature type (number of different values)  
* `split_set` :  
    a reference to store the split criterion  
* `right_stat` :  
    the statistics of the reponses of all remaining data points  
* `min_samples_in_child` :  
    smallest acceptable number of distinct data points in any of the children  
* `min_weight_in_child` :  
    smallest acceptable weight in any of the children  
* `rng` :  
    an pseudo random number generator instance  

Returns
-------
float the loss of this split  

>assumes that the features for categoricals have been properly rounded so
casting them to ints results in the right value!  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::print_info "
`print_info() const`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::latex_representation "
`latex_representation() const -> std::string`  

member function to create a string representing the split criterion  

Returns
-------
std::string a label that characterizes the split  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::get_feature_index "
`get_feature_index() const -> index_t`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::get_num_split_value "
`get_num_split_value() const -> num_t`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::get_cat_split_set "
`get_cat_split_set() const -> std::bitset< max_num_categories >`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::compute_subspaces "
`compute_subspaces(const std::vector< std::vector< num_t > > &subspace) const ->
    std::array< std::vector< std::vector< num_t > >, 2 >`  
";

%feature("docstring") rfr::splits::binary_split_one_feature_rss_loss::can_be_split "
`can_be_split(const std::vector< num_t > &feature_vector) const -> bool`  
";

// File: classrfr_1_1forests_1_1classification__forest.xml


%feature("docstring") rfr::forests::classification_forest "
";

%feature("docstring") rfr::forests::classification_forest::classification_forest "
`classification_forest(forest_options< num_type, response_type, index_type >
    forest_opts)`  
";

%feature("docstring") rfr::forests::classification_forest::fit "
`fit(const rfr::data_containers::data_container_base< num_type, response_type,
    index_type > &data, rng_type &rng)`  
";

%feature("docstring") rfr::forests::classification_forest::predict_class "
`predict_class(num_type *feature_vector) -> std::tuple< num_type, num_type >`  
";

%feature("docstring") rfr::forests::classification_forest::save_latex_representation "
`save_latex_representation(const char *filename_template)`  
";

%feature("docstring") rfr::forests::classification_forest::print_info "
`print_info()`  
";

// File: structrfr_1_1splits_1_1data__info__t.xml


%feature("docstring") rfr::splits::data_info_t "

Attributes
----------
* `index` : `index_t`  

* `response` : `response_t`  

* `prediction_value` : `response_t`  

* `feature` : `num_t`  

* `weight` : `num_t`  
";

%feature("docstring") rfr::splits::data_info_t::data_info_t "
`data_info_t()`  
";

%feature("docstring") rfr::splits::data_info_t::data_info_t "
`data_info_t(index_t i, response_t r, response_t p, num_t f, num_t w)`  
";

// File: classrfr_1_1data__containers_1_1default__container.xml


%feature("docstring") rfr::data_containers::default_container "

A data container for mostly continuous data.  

It might happen that only a small fraction of all features is categorical. In
that case it would be wasteful to store the type of every feature separately.
Instead, this data_container only stores the non-continuous ones in a hash-map.  

C++ includes: default_data_container.hpp
";

%feature("docstring") rfr::data_containers::default_container::default_container "
`default_container(index_t num_f)`  
";

%feature("docstring") rfr::data_containers::default_container::init_protected "
`init_protected(index_t num_f)`  
";

%feature("docstring") rfr::data_containers::default_container::feature "
`feature(index_t feature_index, index_t sample_index) const -> num_t`  
";

%feature("docstring") rfr::data_containers::default_container::features "
`features(index_t feature_index, const std::vector< index_t > &sample_indices)
    const -> std::vector< num_t >`  
";

%feature("docstring") rfr::data_containers::default_container::response "
`response(index_t sample_index) const -> response_t`  
";

%feature("docstring") rfr::data_containers::default_container::predict_value "
`predict_value(index_t sample_index) const -> response_t`  
";

%feature("docstring") rfr::data_containers::default_container::add_data_point "
`add_data_point(std::vector< num_t > features, response_t response, num_t
    weight=1)`  
";

%feature("docstring") rfr::data_containers::default_container::add_data_point "
`add_data_point(std::vector< num_t > features, std::vector< response_t >
    response, num_t weight=1)`  
";

%feature("docstring") rfr::data_containers::default_container::retrieve_data_point "
`retrieve_data_point(index_t index) const -> std::vector< num_t >`  
";

%feature("docstring") rfr::data_containers::default_container::weight "
`weight(index_t sample_index) const -> num_t`  
";

%feature("docstring") rfr::data_containers::default_container::get_type_of_feature "
`get_type_of_feature(index_t feature_index) const -> index_t`  

query the type of a feature  

Parameters
----------
* `feature_index` :  
    the index of the feature  

Returns
-------
int type of the feature: 0 - numerical value (float or int); n>0 - categorical
value with n different values {0,1,...,n-1}  

As most features are assumed to be numerical, it is actually beneficial to store
only the categorical exceptions in a hash-map. Type = 0 means continuous, and
Type = n >= 1 means categorical with options in {0, n-1}.  

Parameters
----------
* `feature_index` :  
    the index of the feature  

Returns
-------
int type of the feature: 0 - numerical value (float or int); n>0 - categorical
value with n different values {1,2,...,n}  
";

%feature("docstring") rfr::data_containers::default_container::set_type_of_feature "
`set_type_of_feature(index_t index, index_t type)`  
";

%feature("docstring") rfr::data_containers::default_container::num_features "
`num_features() const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container::num_data_points "
`num_data_points() const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container::get_type_of_response "
`get_type_of_response() const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container::set_type_of_response "
`set_type_of_response(index_t resp_t)`  
";

%feature("docstring") rfr::data_containers::default_container::set_bounds_of_feature "
`set_bounds_of_feature(index_t feature_index, num_t min, num_t max)`  
";

%feature("docstring") rfr::data_containers::default_container::get_bounds_of_feature "
`get_bounds_of_feature(index_t feature_index) const -> std::pair< num_t, num_t
    >`  
";

%feature("docstring") rfr::data_containers::default_container::get_min_max_of_feature "
`get_min_max_of_feature(index_t feature_index) const -> std::pair< num_t, num_t
    >`  
";

%feature("docstring") rfr::data_containers::default_container::guess_bounds_from_data "
`guess_bounds_from_data()`  
";

%feature("docstring") rfr::data_containers::default_container::normalize_data "
`normalize_data()`  
";

%feature("docstring") rfr::data_containers::default_container::import_csv_files "
`import_csv_files(const std::string &feature_file, const std::string
    &response_file, std::string weight_file=\"\") -> int`  
";

%feature("docstring") rfr::data_containers::default_container::check_consistency "
`check_consistency() -> bool`  
";

%feature("docstring") rfr::data_containers::default_container::print_data "
`print_data()`  
";

// File: classrfr_1_1data__containers_1_1default__container__with__instances.xml


%feature("docstring") rfr::data_containers::default_container_with_instances "

A data container for mostly continuous data with instances.  

Similar to the mostly_continuous_data container, but with the capability to
handle instance features.  

C++ includes: default_data_container_with_instances.hpp
";

%feature("docstring") rfr::data_containers::default_container_with_instances::default_container_with_instances "
`default_container_with_instances()`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::default_container_with_instances "
`default_container_with_instances(index_t num_config_f, index_t num_instance_f)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::feature "
`feature(index_t feature_index, index_t sample_index) const -> num_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::features "
`features(index_t feature_index, const std::vector< index_t > &sample_indices)
    const -> std::vector< num_t >`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::response "
`response(index_t sample_index) const -> response_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::predict_value "
`predict_value(index_t sample_index) const -> response_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::add_data_point "
`add_data_point(std::vector< num_t >, response_t, num_t)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::add_data_point "
`add_data_point(std::vector< num_t >, std::vector< response_t >, num_t)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::add_data_point "
`add_data_point(index_t config_index, index_t instance_index, response_t r,
    num_t weight=1)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::weight "
`weight(index_t sample_index) const -> num_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::num_configurations "
`num_configurations() -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::num_instances "
`num_instances() -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::add_configuration "
`add_configuration(const std::vector< num_t > &config_features) -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::add_instance "
`add_instance(const std::vector< num_t > instance_features) -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::retrieve_data_point "
`retrieve_data_point(index_t index) const -> std::vector< num_t >`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::get_type_of_feature "
`get_type_of_feature(index_t feature_index) const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::set_type_of_configuration_feature "
`set_type_of_configuration_feature(index_t index, index_t type)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::set_type_of_instance_feature "
`set_type_of_instance_feature(index_t index, index_t type)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::set_type_of_feature "
`set_type_of_feature(index_t index, index_t type)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::num_features "
`num_features() const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::num_data_points "
`num_data_points() const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::check_consistency "
`check_consistency()`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::get_type_of_response "
`get_type_of_response() const -> index_t`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::set_type_of_response "
`set_type_of_response(index_t resp_t)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::set_bounds_of_feature "
`set_bounds_of_feature(index_t feature_index, num_t min, num_t max)`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::get_bounds_of_feature "
`get_bounds_of_feature(index_t feature_index) const -> std::pair< num_t, num_t
    >`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::get_instance_set "
`get_instance_set() -> std::vector< num_t >`  

method to get instance as set_feature for
predict_mean_var_of_mean_response_on_set method in regression forest  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::get_configuration_set "
`get_configuration_set(num_t configuration_index) -> std::vector< num_t >`  
";

%feature("docstring") rfr::data_containers::default_container_with_instances::get_features_by_configuration_and_instance "
`get_features_by_configuration_and_instance(num_t configuration_index, num_t
    instance_index) -> std::vector< num_t >`  
";

// File: classrfr_1_1forests_1_1f_a_n_o_v_a__forest.xml


%feature("docstring") rfr::forests::fANOVA_forest "
";

%feature("docstring") rfr::forests::fANOVA_forest::fANOVA_forest "
`fANOVA_forest()`  
";

%feature("docstring") rfr::forests::fANOVA_forest::fANOVA_forest "
`fANOVA_forest(forest_options< num_t, response_t, index_t > forest_opts)`  
";

%feature("docstring") rfr::forests::fANOVA_forest::~fANOVA_forest "
`~fANOVA_forest()`  
";

%feature("docstring") rfr::forests::fANOVA_forest::serialize "
`serialize(Archive &archive)`  

serialize function for saving forests with cerial  
";

%feature("docstring") rfr::forests::fANOVA_forest::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data, rng_t
    &rng)`  
";

%feature("docstring") rfr::forests::fANOVA_forest::set_cutoffs "
`set_cutoffs(num_t lower, num_t upper)`  
";

%feature("docstring") rfr::forests::fANOVA_forest::get_cutoffs "
`get_cutoffs() -> std::pair< num_t, num_t >`  
";

%feature("docstring") rfr::forests::fANOVA_forest::precompute_marginals "
`precompute_marginals()`  
";

%feature("docstring") rfr::forests::fANOVA_forest::marginal_mean_prediction "
`marginal_mean_prediction(const std::vector< num_t > &feature_vector) -> num_t`  
";

%feature("docstring") rfr::forests::fANOVA_forest::marginal_mean_variance_prediction "
`marginal_mean_variance_prediction(const std::vector< num_t > &feature_vector)
    -> std::pair< num_t, num_t >`  
";

%feature("docstring") rfr::forests::fANOVA_forest::marginal_prediction_stat_of_tree "
`marginal_prediction_stat_of_tree(index_t tree_index, const std::vector< num_t >
    &feature_vector) -> rfr::util::weighted_running_statistics< num_t >`  
";

%feature("docstring") rfr::forests::fANOVA_forest::get_trees_total_variances "
`get_trees_total_variances() -> std::vector< num_t >`  
";

%feature("docstring") rfr::forests::fANOVA_forest::all_split_values "
`all_split_values() -> std::vector< std::vector< std::vector< num_t > > >`  
";

// File: structrfr_1_1forests_1_1forest__options.xml


%feature("docstring") rfr::forests::forest_options "

Attributes
----------
* `num_trees` : `index_t`  
    number of trees in the forest  

* `num_data_points_per_tree` : `index_t`  
    number of datapoints used in each tree  

* `do_bootstrapping` : `bool`  
    flag to toggle bootstrapping  

* `compute_oob_error` : `bool`  
    flag to enable/disable computing the out-of-bag error  

* `compute_law_of_total_variance` : `bool`  
    flag to enable/disable computation with the lotv  

* `tree_opts` : `rfr::trees::tree_options< num_t, response_t, index_t >`  
    the options for each tree  
";

%feature("docstring") rfr::forests::forest_options::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::forests::forest_options::set_default_values "
`set_default_values()`  

(Re)set to default values for the forest.  
";

%feature("docstring") rfr::forests::forest_options::adjust_limits_to_data "
`adjust_limits_to_data(const rfr::data_containers::base< num_t, response_t,
    index_t > &data)`  

adjusts all relevant variables to the data  
";

%feature("docstring") rfr::forests::forest_options::forest_options "
`forest_options()`  

Default constructor that initializes the values with their default  
";

%feature("docstring") rfr::forests::forest_options::forest_options "
`forest_options(rfr::trees::tree_options< num_t, response_t, index_t > &to)`  

Constructor to feed in tree values but leave the forest parameters at their
default.  
";

%feature("docstring") rfr::forests::forest_options::forest_options "
`forest_options(rfr::trees::tree_options< num_t, response_t, index_t > &to,
    rfr::data_containers::base< num_t, response_t, index_t > &data)`  

Constructor that adjusts to the data.  
";

%feature("docstring") rfr::forests::forest_options::to_string "
`to_string() const -> std::string`  
";

// File: classrfr_1_1nodes_1_1k__ary__mondrian__node__full.xml


%feature("docstring") rfr::nodes::k_ary_mondrian_node_full "
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::~k_ary_mondrian_node_full "
`~k_ary_mondrian_node_full()`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::k_ary_mondrian_node_full "
`k_ary_mondrian_node_full()`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::k_ary_mondrian_node_full "
`k_ary_mondrian_node_full(int parent, index_t depth, std::array< typename
    std::vector< index_t >::iterator, 3 > info_split)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::k_ary_mondrian_node_full "
`k_ary_mondrian_node_full(int parent, index_t depth, std::array< index_t, 3 >
    info_split_index)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::k_ary_mondrian_node_full "
`k_ary_mondrian_node_full(int parent, index_t depth)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_sum_of_Min_Max_intervals "
`get_sum_of_Min_Max_intervals() const -> const num_t`  

get reference to the response values  

get the sum of the mx-min intervals fo the node  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_parent_index "
`get_parent_index() const -> const int`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_split_time "
`get_split_time() const -> const num_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_split_dimension "
`get_split_dimension() const -> const index_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_split_value "
`get_split_value() const -> const num_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_min_max "
`get_min_max() const -> const std::vector< std::pair< num_t, num_t > >`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_info_split_its "
`get_info_split_its() const -> const std::array< typename std::vector< index_t
    >::iterator, 3 >`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_info_split_its_index "
`get_info_split_its_index() const -> const std::array< index_t, 3 >`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_variance "
`get_variance() const -> const num_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_mean "
`get_mean() const -> const num_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_number_of_points "
`get_number_of_points() const -> const int`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::get_split_cost "
`get_split_cost() const -> const num_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_sum_of_Min_Max_intervals "
`set_sum_of_Min_Max_intervals(num_t sum)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_parent_index "
`set_parent_index(index_t parent)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_split_time "
`set_split_time(num_t time_s)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_split_dimension "
`set_split_dimension(index_t split_d)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_split_value "
`set_split_value(num_t split_v)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_min_max "
`set_min_max(std::vector< std::pair< num_t, num_t >> m_m)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_info_split_its "
`set_info_split_its(std::array< typename std::vector< index_t >::iterator, 3 >
    info_split)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_info_split_its_index "
`set_info_split_its_index(std::array< index_t, 3 > info_split)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_variance "
`set_variance(num_t var)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_mean "
`set_mean(num_t m)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_number_of_points "
`set_number_of_points(int p)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::set_split_cost "
`set_split_cost(num_t sc)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::add_response "
`add_response(response_t response, num_t weight)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_full::print_info "
`print_info() const`  

prints out some basic information about the node  
";

// File: classrfr_1_1nodes_1_1k__ary__mondrian__node__minimal.xml


%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal "
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::~k_ary_mondrian_node_minimal "
`~k_ary_mondrian_node_minimal()`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::k_ary_mondrian_node_minimal "
`k_ary_mondrian_node_minimal()`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::k_ary_mondrian_node_minimal "
`k_ary_mondrian_node_minimal(index_t level)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::is_a_leaf "
`is_a_leaf() const -> bool`  

to test whether this node is a leaf  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::get_children "
`get_children() const -> std::array< index_t, k >`  

get the index of the node's parent  

get indices of all children  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::get_child_index "
`get_child_index(index_t idx) const -> index_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::get_depth "
`get_depth() const -> index_t`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::get_response_stat "
`get_response_stat() const -> rfr::util::weighted_running_statistics< num_t >`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::falls_into_child "
`falls_into_child(const std::vector< num_t > &feature_vector) const -> index_t`  

returns the index of the child into which the provided sample falls  

Parameters
----------
* `feature_vector` :  
    a feature vector of the appropriate size (not checked!)  

Returns
-------
index_t index of the child  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::set_child "
`set_child(index_t idx, index_t child)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::set_depth "
`set_depth(index_t new_depth)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::set_response_stat "
`set_response_stat(rfr::util::weighted_running_statistics< num_t > r_s)`  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::print_info "
`print_info() const`  

prints out some basic information about the node  
";

%feature("docstring") rfr::nodes::k_ary_mondrian_node_minimal::latex_representation "
`latex_representation(int my_index) const -> std::string`  

generates a label for the node to be used in the LaTeX visualization  
";

// File: classrfr_1_1trees_1_1k__ary__mondrian__tree.xml


%feature("docstring") rfr::trees::k_ary_mondrian_tree "
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::k_ary_mondrian_tree "
`k_ary_mondrian_tree()`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::~k_ary_mondrian_tree "
`~k_ary_mondrian_tree()`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::set_tree_options "
`set_tree_options(rfr::trees::tree_options< num_t, response_t, index_t >
    tree_opts)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::partial_fit "
`partial_fit(const rfr::data_containers::base< num_t, response_t, index_t >
    &data, rfr::trees::tree_options< num_t, response_t, index_t > tree_opts,
    index_t new_point, rng_t &rng)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::internal_partial_fit "
`internal_partial_fit(const rfr::data_containers::base< num_t, response_t,
    index_t > &data, rfr::trees::tree_options< num_t, response_t, index_t >
    tree_opts, index_t new_point, rng_t &rng)`  

internal_partial_fit adds a point to the current mondrian tree  

Finds the place of the new_point in the tree and adds the point in that part of
the tree.  

made. Just make sure the max_features in tree_opts to a number smaller than the
number of features!  

Parameters
----------
* `data` :  
    the container holding the training data  
* `tree_opts` :  
    a tree_options object that controls certain aspects of \"growing\" the tree  
* `new_point` :  
    index of the point to add  
* `rng` :  
    the random number generator to be used  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::update_subtree "
`update_subtree(index_t start)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::addNewNode "
`addNewNode(rfr::nodes::k_ary_mondrian_node_full< k, num_t, response_t, index_t,
    rng_t > new_node, int initial_position, bool adding_parent)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data,
    rfr::trees::tree_options< num_t, response_t, index_t > tree_opts, const
    std::vector< num_t > &sample_weights, rng_t &rng)`  

fits a randomized decision tree to a subset of the data  

At each node, if it is 'splitworthy', a random subset of all features is
considered for the split. Depending on the split_type provided, greedy or
randomized choices can be made. Just make sure the max_features in tree_opts to
a number smaller than the number of features!  

Parameters
----------
* `data` :  
    the container holding the training data  
* `tree_opts` :  
    a tree_options object that controls certain aspects of \"growing\" the tree  
* `sample_weights` :  
    vector containing the weights of all allowed datapoints (set to individual
    entries to zero for subsampling), no checks are done here!  
* `rng` :  
    the random number generator to be used  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::Sample_Mondrian_Block "
`Sample_Mondrian_Block(const rfr::data_containers::base< num_t, response_t,
    index_t > &data, std::vector< rfr::nodes::k_ary_mondrian_node_full< k,
    num_t, response_t, index_t, rng_t >> &tmp_nodes, std::vector< index_t >
    &selected_elements, std::vector< response_t > responses, index_t position,
    rng_t &rng) -> rfr::nodes::k_ary_mondrian_node_full< k, num_t, response_t,
    index_t, rng_t >`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::internal_fit "
`internal_fit(const rfr::data_containers::base< num_t, response_t, index_t >
    &data, rfr::trees::tree_options< num_t, response_t, index_t > tree_opts,
    const std::vector< num_t > &sample_weights, rng_t &rng)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::update_gaussian_parameters "
`update_gaussian_parameters(const rfr::data_containers::base< num_t, response_t,
    index_t > &data)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::update_likelyhood "
`update_likelyhood()`  

noise precision  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::get_parent_split_time "
`get_parent_split_time(rfr::nodes::k_ary_mondrian_node_full< k, num_t,
    response_t, index_t, rng_t > node) -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::get_sigmoid_variance "
`get_sigmoid_variance(index_t node_index) -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::sigmoid "
`sigmoid(num_t x) -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::multiply_gausian "
`multiply_gausian(std::pair< response_t, response_t > g1, std::pair< response_t,
    response_t > g2) -> std::pair< response_t, response_t >`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::myPartition "
`myPartition(index_t it1, index_t it2, std::vector< index_t >
    &selected_elements, const rfr::data_containers::base< num_t, response_t,
    index_t > &data, index_t split_dimension, num_t split_value) -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::min_max_vector "
`min_max_vector(const rfr::data_containers::base< num_t, response_t, index_t >
    &data, const std::array< index_t, 3 > &its, const std::vector< index_t >
    &selected_elements, num_t &sum_E) -> std::vector< std::pair< num_t, num_t >
    >`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::min_max_feature "
`min_max_feature(const std::vector< num_t > feature_values, num_t &min, num_t
    &max, num_t &sum_E)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::find_leaf_index "
`find_leaf_index(const std::vector< num_t > &feature_vector) const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::get_leaf "
`get_leaf(const std::vector< num_t > &feature_vector) const -> const node_t &`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::predict_mean_var "
`predict_mean_var(const std::vector< num_t > &feature_vector) -> std::pair<
    num_t, num_t >`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::predict "
`predict(const std::vector< num_t > &feature_vector) const -> response_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::calculate_nu "
`calculate_nu(rfr::nodes::k_ary_mondrian_node_full< k, num_t, response_t,
    index_t, rng_t > &tmp_node, const std::vector< num_t > &feature_vector)
    const -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::marginalized_mean_prediction "
`marginalized_mean_prediction(const std::vector< num_t > &feature_vector,
    index_t node_index=0) const -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::leaf_entries "
`leaf_entries(const std::vector< num_t > &feature_vector) const -> const
    std::vector< response_t > &`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::all_split_values "
`all_split_values(const std::vector< index_t > &types) const -> std::vector<
    std::vector< num_t > >`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::number_of_nodes "
`number_of_nodes() const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::number_of_leafs "
`number_of_leafs() const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::depth "
`depth() const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::partition_recursor "
`partition_recursor(std::vector< std::vector< std::vector< num_t > > >
    &the_partition, std::vector< std::vector< num_t > > &subspace, num_t
    node_index) const`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::partition "
`partition(std::vector< std::vector< num_t > > pcs) const -> std::vector<
    std::vector< std::vector< num_t > > >`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::total_weight_in_subtree "
`total_weight_in_subtree(index_t node_index) const -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::check_split_fractions "
`check_split_fractions(num_t epsilon=1e-6) const -> bool`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::pseudo_update "
`pseudo_update(std::vector< num_t > features, response_t response, num_t
    weight)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::pseudo_downdate "
`pseudo_downdate(std::vector< num_t > features, response_t response, num_t
    weight)`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::max_life_time "
`max_life_time() const -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::print_info "
`print_info() const`  
";

%feature("docstring") rfr::trees::k_ary_mondrian_tree::save_latex_representation "
`save_latex_representation(const char *filename) const`  

a visualization by generating a LaTeX document that can be compiled  

Parameters
----------
* `filename` :  
    Name of the file that will be used. Note that any existing file will be
    silently overwritten!  
";

// File: classrfr_1_1nodes_1_1k__ary__node__full.xml


%feature("docstring") rfr::nodes::k_ary_node_full "

The node class for regular k-ary trees.  

In a regular k-ary tree, every node has either zero (a leaf) or exactly
k-children (an internal node). In this case, one can try to gain some speed by
replacing variable length std::vectors by std::arrays.  

C++ includes: k_ary_node.hpp
";

%feature("docstring") rfr::nodes::k_ary_node_full::~k_ary_node_full "
`~k_ary_node_full()`  
";

%feature("docstring") rfr::nodes::k_ary_node_full::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::nodes::k_ary_node_full::push_response_value "
`push_response_value(response_t r, num_t w)`  

adds an observation to the leaf node  

This function can be used for pseudo updates of a tree by simply adding
observations into the corresponding leaf  
";

%feature("docstring") rfr::nodes::k_ary_node_full::pop_response_value "
`pop_response_value(response_t r, num_t w)`  

removes the last added observation from the leaf node  

This function can be used for pseudo updates of a tree by simply adding
observations into the corresponding leaf  

Parameters
----------
* `r` :  
    ignored  
* `w` :  
    ignored  
";

%feature("docstring") rfr::nodes::k_ary_node_full::responses "
`responses() const -> const std::vector< response_t > &`  

get reference to the response values  
";

%feature("docstring") rfr::nodes::k_ary_node_full::weights "
`weights() const -> const std::vector< num_t > &`  

get reference to the response values  
";

%feature("docstring") rfr::nodes::k_ary_node_full::print_info "
`print_info() const`  

prints out some basic information about the node  
";

// File: classrfr_1_1nodes_1_1k__ary__node__minimal.xml


%feature("docstring") rfr::nodes::k_ary_node_minimal "
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::~k_ary_node_minimal "
`~k_ary_node_minimal()`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::make_internal_node "
`make_internal_node(const rfr::nodes::temporary_node< num_t, response_t, index_t
    > &tmp_node, const rfr::data_containers::base< num_t, response_t, index_t >
    &data, std::vector< index_t > &features_to_try, index_t num_nodes,
    std::deque< rfr::nodes::temporary_node< num_t, response_t, index_t > >
    &tmp_nodes, index_t min_samples_in_leaf, num_t min_weight_in_leaf, rng_t
    &rng) -> num_t`  

If the temporary node should be split further, this member turns this node into
an internal node.  

Parameters
----------
* `tmp_node` :  
    a temporary_node struct containing all the important information. It is not
    changed in this function.  
* `data` :  
    a refernce to the data object that is used  
* `features_to_try` :  
    vector of allowed features to be used for this split  
* `num_nodes` :  
    number of already created nodes  
* `tmp_nodes` :  
    a deque instance containing all temporary nodes that still have to be
    checked  
* `min_samples_in_leaf` :  
    sets the minimum number of distinct data points in a leaf  
* `min_weight_in_leaf` :  
    sets the minimum sum of sample weights in a leaf  
* `rng` :  
    a RNG instance  

Returns
-------
num_t the loss of the split  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::make_leaf_node "
`make_leaf_node(const rfr::nodes::temporary_node< num_t, response_t, index_t >
    &tmp_node, const rfr::data_containers::base< num_t, response_t, index_t >
    &data)`  

turns this node into a leaf node based on a temporary node.  

Parameters
----------
* `tmp_node` :  
    the internal representation for a temporary node.  
* `data` :  
    a data container instance  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::can_be_split "
`can_be_split(const std::vector< num_t > &feature_vector) const -> bool`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::falls_into_child "
`falls_into_child(const std::vector< num_t > &feature_vector) const -> index_t`  

returns the index of the child into which the provided sample falls  

Parameters
----------
* `feature_vector` :  
    a feature vector of the appropriate size (not checked!)  

Returns
-------
index_t index of the child  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::push_response_value "
`push_response_value(response_t r, num_t w)`  

adds an observation to the leaf node  

This function can be used for pseudo updates of a tree by simply adding
observations into the corresponding leaf  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::pop_response_value "
`pop_response_value(response_t r, num_t w)`  

removes an observation from the leaf node  

This function can be used for pseudo updates of a tree by simply removing
observations from the corresponding leaf  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::compute_subspaces "
`compute_subspaces(const std::vector< std::vector< num_t > > &subspace) const ->
    std::array< std::vector< std::vector< num_t > >, 2 >`  

helper function for the fANOVA  

See description of
rfr::splits::binary_split_one_feature_rss_loss::compute_subspace.  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::leaf_statistic "
`leaf_statistic() const -> const rfr::util::weighted_running_statistics< num_t >
    &`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::is_a_leaf "
`is_a_leaf() const -> bool`  

to test whether this node is a leaf  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::parent "
`parent() const -> index_t`  

get the index of the node's parent  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::get_children "
`get_children() const -> std::array< index_t, k >`  

get indices of all children  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::get_child_index "
`get_child_index(index_t idx) const -> index_t`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::get_split_fractions "
`get_split_fractions() const -> std::array< num_t, k >`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::get_split_fraction "
`get_split_fraction(index_t idx) const -> num_t`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::get_split "
`get_split() const -> const split_type &`  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::print_info "
`print_info() const`  

prints out some basic information about the node  
";

%feature("docstring") rfr::nodes::k_ary_node_minimal::latex_representation "
`latex_representation(int my_index) const -> std::string`  

generates a label for the node to be used in the LaTeX visualization  
";

// File: classrfr_1_1trees_1_1k__ary__random__tree.xml


%feature("docstring") rfr::trees::k_ary_random_tree "
";

%feature("docstring") rfr::trees::k_ary_random_tree::k_ary_random_tree "
`k_ary_random_tree()`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::~k_ary_random_tree "
`~k_ary_random_tree()`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data,
    rfr::trees::tree_options< num_t, response_t, index_t > tree_opts, const
    std::vector< num_t > &sample_weights, rng_type &rng)`  

fits a randomized decision tree to a subset of the data  

At each node, if it is 'splitworthy', a random subset of all features is
considered for the split. Depending on the split_type provided, greedy or
randomized choices can be made. Just make sure the max_features in tree_opts to
a number smaller than the number of features!  

Parameters
----------
* `data` :  
    the container holding the training data  
* `tree_opts` :  
    a tree_options object that controls certain aspects of \"growing\" the tree  
* `sample_weights` :  
    vector containing the weights of all allowed datapoints (set to individual
    entries to zero for subsampling), no checks are done here!  
* `rng` :  
    the random number generator to be used  
";

%feature("docstring") rfr::trees::k_ary_random_tree::find_leaf_index "
`find_leaf_index(const std::vector< num_t > &feature_vector) const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::get_leaf "
`get_leaf(const std::vector< num_t > &feature_vector) const -> const node_type
    &`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::leaf_entries "
`leaf_entries(const std::vector< num_t > &feature_vector) const -> const
    std::vector< response_t > &`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::leaf_statistic "
`leaf_statistic(const std::vector< num_t > &feature_vector) const -> const
    rfr::util::weighted_running_statistics< num_t > &`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::predict "
`predict(const std::vector< num_t > &feature_vector) const -> response_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::marginalized_mean_prediction "
`marginalized_mean_prediction(const std::vector< num_t > &feature_vector,
    index_t node_index=0) const -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::number_of_nodes "
`number_of_nodes() const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::number_of_leafs "
`number_of_leafs() const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::depth "
`depth() const -> index_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::partition_recursor "
`partition_recursor(std::vector< std::vector< std::vector< num_t > > >
    &the_partition, std::vector< std::vector< num_t > > &subspace, num_t
    node_index) const`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::partition "
`partition(std::vector< std::vector< num_t > > pcs) const -> std::vector<
    std::vector< std::vector< num_t > > >`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::total_weight_in_subtree "
`total_weight_in_subtree(index_t node_index) const -> num_t`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::check_split_fractions "
`check_split_fractions(num_t epsilon=1e-6) const -> bool`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::pseudo_update "
`pseudo_update(std::vector< num_t > features, response_t response, num_t
    weight)`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::pseudo_downdate "
`pseudo_downdate(std::vector< num_t > features, response_t response, num_t
    weight)`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::print_info "
`print_info() const`  
";

%feature("docstring") rfr::trees::k_ary_random_tree::save_latex_representation "
`save_latex_representation(const char *filename) const`  

a visualization by generating a LaTeX document that can be compiled  

Parameters
----------
* `filename` :  
    Name of the file that will be used. Note that any existing file will be
    silently overwritten!  
";

// File: classrfr_1_1splits_1_1k__ary__split__base.xml


%feature("docstring") rfr::splits::k_ary_split_base "
";

%feature("docstring") rfr::splits::k_ary_split_base::~k_ary_split_base "
`~k_ary_split_base()`  
";

%feature("docstring") rfr::splits::k_ary_split_base::find_best_split "
`find_best_split(const rfr::data_containers::base< num_t, response_t, index_t >
    &data, const std::vector< index_t > &features_to_try, typename std::vector<
    data_info_t< num_t, response_t, index_t > >::iterator infos_begin, typename
    std::vector< data_info_t< num_t, response_t, index_t > >::iterator
    infos_end, std::array< typename std::vector< data_info_t< num_t, response_t,
    index_t > >::iterator, k+1 > &info_split_its, index_t min_samples_in_child,
    num_t min_weight_in_child, rng_t &rng)=0 -> num_t`  

member function to find the optimal split for a subset of the data and features  

Defining the interface that every split has to implement. Unfortunately, virtual
constructors are not allowed in C++, so this function is called instead. Code in
the nodes and the tree will only use the default constructor and the methods
below for training and prediction.  

Parameters
----------
* `data` :  
    the container holding the training data  
* `features_to_try` :  
    a vector with the indices of all the features that can be considered for
    this split  
* `infos_begin` :  
    iterator to the first data_info element to be considered  
* `infos_end` :  
    iterator beyond the last data_info element to be considered  
* `info_split_its` :  
    iterators into indices specifying where to split the data for the children.
    Number of iterators is k+1, for easier iteration  
* `min_samples_in_child` :  
    smallest acceptable number of samples in any of the children  
* `min_weight_in_child` :  
    smallest acceptable weight in any of the children  
* `rng` :  
    (pseudo) random number generator as a source for stochasticity  

Returns
-------
float the loss of the found split  
";

%feature("docstring") rfr::splits::k_ary_split_base::print_info "
`print_info() const =0`  

some debug output that prints a informative representation to std::cout  
";

%feature("docstring") rfr::splits::k_ary_split_base::latex_representation "
`latex_representation() const =0 -> std::string`  

hopefully all trees can create a LaTeX document as a visualization, this
contributes the text of the split.  
";

// File: classrfr_1_1forests_1_1mondrian__forest.xml


%feature("docstring") rfr::forests::mondrian_forest "

Attributes
----------
* `options` : `forest_options< num_t, response_t, index_t >`  

* `internal_index` : `index_t`  

* `name` : `std::string`  
";

%feature("docstring") rfr::forests::mondrian_forest::get_trees "
`get_trees() const -> std::vector< tree_t >`  
";

%feature("docstring") rfr::forests::mondrian_forest::serialize "
`serialize(Archive &archive)`  

serialize function for saving forests with cerial  
";

%feature("docstring") rfr::forests::mondrian_forest::mondrian_forest "
`mondrian_forest()`  
";

%feature("docstring") rfr::forests::mondrian_forest::mondrian_forest "
`mondrian_forest(forest_options< num_t, response_t, index_t > opts)`  
";

%feature("docstring") rfr::forests::mondrian_forest::~mondrian_forest "
`~mondrian_forest()`  
";

%feature("docstring") rfr::forests::mondrian_forest::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data, rng_t
    &rng)`  

growing the random forest for a given data set  

Parameters
----------
* `data` :  
    a filled data container  
* `rng` :  
    the random number generator to be used  
";

%feature("docstring") rfr::forests::mondrian_forest::predict_mean_var "
`predict_mean_var(const std::vector< num_t > &feature_vector) -> std::pair<
    num_t, num_t >`  
";

%feature("docstring") rfr::forests::mondrian_forest::predict "
`predict(const std::vector< num_t > &feature_vector) const -> response_t`  
";

%feature("docstring") rfr::forests::mondrian_forest::predict_median "
`predict_median(const std::vector< num_t > &feature_vector) -> response_t`  
";

%feature("docstring") rfr::forests::mondrian_forest::partial_fit "
`partial_fit(const rfr::data_containers::base< num_t, response_t, index_t >
    &data, rng_t &rng, index_t point)`  
";

%feature("docstring") rfr::forests::mondrian_forest::out_of_bag_error "
`out_of_bag_error() -> num_t`  
";

%feature("docstring") rfr::forests::mondrian_forest::save_to_binary_file "
`save_to_binary_file(const std::string filename)`  
";

%feature("docstring") rfr::forests::mondrian_forest::load_from_binary_file "
`load_from_binary_file(const std::string filename)`  
";

%feature("docstring") rfr::forests::mondrian_forest::ascii_string_representation "
`ascii_string_representation() -> std::string`  
";

%feature("docstring") rfr::forests::mondrian_forest::load_from_ascii_string "
`load_from_ascii_string(std::string const &str)`  
";

%feature("docstring") rfr::forests::mondrian_forest::save_latex_representation "
`save_latex_representation(const std::string filename_template)`  
";

%feature("docstring") rfr::forests::mondrian_forest::print_info "
`print_info()`  
";

%feature("docstring") rfr::forests::mondrian_forest::num_trees "
`num_trees() -> unsigned int`  
";

// File: classrfr_1_1forests_1_1quantile__regression__forest.xml


%feature("docstring") rfr::forests::quantile_regression_forest "
";

%feature("docstring") rfr::forests::quantile_regression_forest::quantile_regression_forest "
`quantile_regression_forest()`  
";

%feature("docstring") rfr::forests::quantile_regression_forest::quantile_regression_forest "
`quantile_regression_forest(forest_options< num_t, response_t, index_t >
    forest_opts)`  
";

%feature("docstring") rfr::forests::quantile_regression_forest::~quantile_regression_forest "
`~quantile_regression_forest()`  
";

%feature("docstring") rfr::forests::quantile_regression_forest::predict_quantiles "
`predict_quantiles(const std::vector< num_t > &feature_vector, std::vector<
    num_t > quantiles) const -> std::vector< num_t >`  
";

// File: classrfr_1_1forests_1_1regression__forest.xml


%feature("docstring") rfr::forests::regression_forest "

Attributes
----------
* `options` : `forest_options< num_t, response_t, index_t >`  
";

%feature("docstring") rfr::forests::regression_forest::serialize "
`serialize(Archive &archive)`  

serialize function for saving forests with cerial  
";

%feature("docstring") rfr::forests::regression_forest::regression_forest "
`regression_forest()`  
";

%feature("docstring") rfr::forests::regression_forest::regression_forest "
`regression_forest(forest_options< num_t, response_t, index_t > opts)`  
";

%feature("docstring") rfr::forests::regression_forest::~regression_forest "
`~regression_forest()`  
";

%feature("docstring") rfr::forests::regression_forest::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data,
    rng_type &rng)`  

growing the random forest for a given data set  

Parameters
----------
* `data` :  
    a filled data container  
* `rng` :  
    the random number generator to be used  
";

%feature("docstring") rfr::forests::regression_forest::predict "
`predict(const std::vector< num_t > &feature_vector) const -> response_t`  
";

%feature("docstring") rfr::forests::regression_forest::predict_mean_var "
`predict_mean_var(const std::vector< num_t > &feature_vector, bool
    weighted_data=false) -> std::pair< num_t, num_t >`  
";

%feature("docstring") rfr::forests::regression_forest::covariance "
`covariance(const std::vector< num_t > &f1, const std::vector< num_t > &f2) ->
    num_t`  
";

%feature("docstring") rfr::forests::regression_forest::kernel "
`kernel(const std::vector< num_t > &f1, const std::vector< num_t > &f2) ->
    num_t`  
";

%feature("docstring") rfr::forests::regression_forest::all_leaf_values "
`all_leaf_values(const std::vector< num_t > &feature_vector) const ->
    std::vector< std::vector< num_t > >`  
";

%feature("docstring") rfr::forests::regression_forest::pseudo_update "
`pseudo_update(std::vector< num_t > features, response_t response, num_t
    weight)`  
";

%feature("docstring") rfr::forests::regression_forest::pseudo_downdate "
`pseudo_downdate(std::vector< num_t > features, response_t response, num_t
    weight)`  
";

%feature("docstring") rfr::forests::regression_forest::out_of_bag_error "
`out_of_bag_error() -> num_t`  
";

%feature("docstring") rfr::forests::regression_forest::save_to_binary_file "
`save_to_binary_file(const std::string filename)`  
";

%feature("docstring") rfr::forests::regression_forest::load_from_binary_file "
`load_from_binary_file(const std::string filename)`  
";

%feature("docstring") rfr::forests::regression_forest::ascii_string_representation "
`ascii_string_representation() -> std::string`  
";

%feature("docstring") rfr::forests::regression_forest::load_from_ascii_string "
`load_from_ascii_string(std::string const &str)`  
";

%feature("docstring") rfr::forests::regression_forest::save_latex_representation "
`save_latex_representation(const std::string filename_template)`  
";

%feature("docstring") rfr::forests::regression_forest::print_info "
`print_info()`  
";

%feature("docstring") rfr::forests::regression_forest::num_trees "
`num_trees() -> unsigned int`  
";

// File: classrfr_1_1util_1_1running__covariance.xml


%feature("docstring") rfr::util::running_covariance "
";

%feature("docstring") rfr::util::running_covariance::running_covariance "
`running_covariance()`  
";

%feature("docstring") rfr::util::running_covariance::push "
`push(num_t x1, num_t x2)`  
";

%feature("docstring") rfr::util::running_covariance::number_of_points "
`number_of_points() -> long unsigned int`  
";

%feature("docstring") rfr::util::running_covariance::covariance "
`covariance() -> num_t`  
";

// File: classrfr_1_1util_1_1running__statistics.xml


%feature("docstring") rfr::util::running_statistics "

simple class to compute mean and variance sequentially one value at a time  

C++ includes: util.hpp
";

%feature("docstring") rfr::util::running_statistics::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::util::running_statistics::running_statistics "
`running_statistics()`  
";

%feature("docstring") rfr::util::running_statistics::running_statistics "
`running_statistics(long unsigned int n, num_t a, num_t s)`  
";

%feature("docstring") rfr::util::running_statistics::push "
`push(num_t x)`  

adds a value to the statistic  

Parameters
----------
* `x` :  
    the value to add  
";

%feature("docstring") rfr::util::running_statistics::pop "
`pop(num_t x)`  

removes a value from the statistic  

Consider this the inverse operation to push. Note: you can create a scenario
where the variance would be negative, so a simple sanity check is implemented
that raises a RuntimeError if that happens.  

Parameters
----------
* `x` :  
    the value to remove  
";

%feature("docstring") rfr::util::running_statistics::divide_sdm_by "
`divide_sdm_by(num_t value) const -> num_t`  

divides the (summed) squared distance from the mean by the argument  

Parameters
----------
* `value` :  
    to divide by  

Returns
-------
sum([ (x - mean())**2 for x in values])/ value  
";

%feature("docstring") rfr::util::running_statistics::number_of_points "
`number_of_points() const -> long unsigned int`  

returns the number of points  

Returns
-------
the current number of points added  
";

%feature("docstring") rfr::util::running_statistics::mean "
`mean() const -> num_t`  

the mean of all values added  

Returns
-------
sum([x for x in values])/number_of_points()  
";

%feature("docstring") rfr::util::running_statistics::sum "
`sum() const -> num_t`  

the sum of all values added  

Returns
-------
the sum of all values (equivalent to number_of_points()* mean())  
";

%feature("docstring") rfr::util::running_statistics::sum_of_squares "
`sum_of_squares() const -> num_t`  

the sum of all values squared  

Returns
-------
sum([x**2 for x in values])  
";

%feature("docstring") rfr::util::running_statistics::variance_population "
`variance_population() const -> num_t`  

the variance of all samples assuming it is the total population  

Returns
-------
sum([(x-mean())**2 for x in values])/number_of_points  
";

%feature("docstring") rfr::util::running_statistics::variance_sample "
`variance_sample() const -> num_t`  

unbiased variance of all samples assuming it is a sample from a population with
unknown mean  

Returns
-------
sum([(x-mean())**2 for x in values])/(number_of_points-1)  
";

%feature("docstring") rfr::util::running_statistics::variance_MSE "
`variance_MSE() const -> num_t`  

biased estimate variance of all samples with the smalles MSE  

Returns
-------
sum([(x-mean())**2 for x in values])/(number_of_points+1)  
";

%feature("docstring") rfr::util::running_statistics::std_population "
`std_population() const -> num_t`  

standard deviation based on variance_population  

Returns
-------
sqrt(variance_population())  
";

%feature("docstring") rfr::util::running_statistics::std_sample "
`std_sample() const -> num_t`  

(biased) estimate of the standard deviation based on variance_sample  

Returns
-------
sqrt(variance_sample())  
";

%feature("docstring") rfr::util::running_statistics::std_unbiased_gaussian "
`std_unbiased_gaussian() const -> num_t`  

unbiased standard deviation for normally distributed values  

Source: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation  

Returns
-------
std_sample/correction_value  
";

%feature("docstring") rfr::util::running_statistics::numerically_equal "
`numerically_equal(const running_statistics other, num_t rel_error) -> bool`  

method to check for numerical equivalency  

Parameters
----------
* `other` :  
    the other running statistic to compare against  
* `rel_error` :  
    relative tolerance for the mean and variance  
";

// File: structrfr_1_1nodes_1_1temporary__node.xml


%feature("docstring") rfr::nodes::temporary_node "

Attributes
----------
* `node_index` : `index_t`  

* `parent_index` : `index_t`  

* `begin` : `std::vector< rfr::splits::data_info_t< num_t, response_t, index_t > >::iterator`  

* `end` : `std::vector< rfr::splits::data_info_t< num_t, response_t, index_t > >::iterator`  

* `node_level` : `index_t`  
";

%feature("docstring") rfr::nodes::temporary_node::temporary_node "
`temporary_node(index_t node_id, index_t parent_id, index_t node_lvl, typename
    std::vector< rfr::splits::data_info_t< num_t, response_t, index_t
    >>::iterator b, typename std::vector< rfr::splits::data_info_t< num_t,
    response_t, index_t >>::iterator e)`  
";

%feature("docstring") rfr::nodes::temporary_node::total_weight "
`total_weight() -> num_t`  
";

%feature("docstring") rfr::nodes::temporary_node::print_info "
`print_info() const`  
";

// File: classrfr_1_1trees_1_1tree__base.xml


%feature("docstring") rfr::trees::tree_base "
";

%feature("docstring") rfr::trees::tree_base::~tree_base "
`~tree_base()`  
";

%feature("docstring") rfr::trees::tree_base::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data,
    rfr::trees::tree_options< num_t, response_t, index_t > tree_opts, rng_type
    &rng)`  

member function to fit the tree to the whole data  

The interface is very general, and allows for deterministic and randomized
decision tree at this point. For a random forest, some randomness has to be
introduced, and the number of features considered for every step has to be set
to be less than actual the number. In its standard implementation this function
just calls the second fit method with an indicex vector = [0, ...,
num_data_points-1].  

Parameters
----------
* `data` :  
    the container holding the training data  
* `tree_opts` :  
    a tree_options opject that controls certain aspects of \"growing\" the tree  
* `rng` :  
    a (pseudo) random number generator  
";

%feature("docstring") rfr::trees::tree_base::fit "
`fit(const rfr::data_containers::base< num_t, response_t, index_t > &data,
    rfr::trees::tree_options< num_t, response_t, index_t > tree_opts, const
    std::vector< num_t > &sample_weights, rng_type &rng)=0`  

fits a (possibly randomized) decision tree to a subset of the data  

At each node, if it is 'splitworthy', a random subset of all features is
considered for the split. Depending on the split_type provided, greedy or
randomized choices can be made. Just make sure the max_features in tree_opts to
a number smaller than the number of features!  

Parameters
----------
* `data` :  
    the container holding the training data  
* `tree_opts` :  
    a tree_options opject that controls certain aspects of \"growing\" the tree  
* `sample_weights` :  
    vector containing the weights of all datapoints, can be used for subsampling
    (no checks are done here!)  
* `rng` :  
    a (pseudo) random number generator  
";

%feature("docstring") rfr::trees::tree_base::predict "
`predict(const std::vector< num_t > &feature_vector) const =0 -> response_t`  

predicts the response value for a single feature vector  

Parameters
----------
* `feature_vector` :  
    an array containing a valid (in terms of size and values!) feature vector  

Returns
-------
num_t the prediction of the response value (usually the mean of all responses in
the corresponding leaf)  
";

%feature("docstring") rfr::trees::tree_base::leaf_entries "
`leaf_entries(const std::vector< num_t > &feature_vector) const =0 -> const
    std::vector< response_t > &`  

returns all response values in the leaf into which the given feature vector
falls  

Parameters
----------
* `feature_vector` :  
    an array containing a valid (in terms of size and values!) feature vector  

Returns
-------
std::vector<response_t> all response values in that leaf  
";

%feature("docstring") rfr::trees::tree_base::number_of_nodes "
`number_of_nodes() const =0 -> index_t`  
";

%feature("docstring") rfr::trees::tree_base::number_of_leafs "
`number_of_leafs() const =0 -> index_t`  
";

%feature("docstring") rfr::trees::tree_base::depth "
`depth() const =0 -> index_t`  
";

%feature("docstring") rfr::trees::tree_base::save_latex_representation "
`save_latex_representation(const char *filename) const =0`  

creates a LaTeX document visualizing the tree  
";

// File: structrfr_1_1trees_1_1tree__options.xml


%feature("docstring") rfr::trees::tree_options "

Attributes
----------
* `max_features` : `index_t`  
    number of features to consider for each split  

* `max_depth` : `index_t`  
    maximum depth for the tree  

* `min_samples_to_split` : `index_t`  
    minumum number of samples to try splitting  

* `min_weight_to_split` : `num_t`  
    minumum weight of samples to try splitting  

* `min_samples_in_leaf` : `index_t`  
    minimum total sample weights in a leaf  

* `min_weight_in_leaf` : `num_t`  
    minimum total sample weights in a leaf  

* `max_num_nodes` : `index_t`  
    maxmimum total number of nodes in the tree  

* `max_num_leaves` : `index_t`  
    maxmimum total number of leaves in the tree  

* `epsilon_purity` : `response_t`  
    minimum difference between two response values to be considered different*/  

* `life_time` : `num_t`  
    life time of a mondrian tree  

* `hierarchical_smoothing` : `bool`  
    flag to enable/disable hierachical smoothing for mondrian forests  
";

%feature("docstring") rfr::trees::tree_options::serialize "
`serialize(Archive &archive)`  

serialize function for saving forests  
";

%feature("docstring") rfr::trees::tree_options::set_default_values "
`set_default_values()`  

(Re)set to default values with no limits on the size of the tree  

If nothing is know about the data, this member can be used to get a valid
setting for the tree_options struct. But beware this setting could lead to a
huge tree depending on the amount of data. There is no limit to the size, and
nodes are split into pure leafs. For each split, every feature is considered!
This not only slows the training down, but also makes this tree deterministic!  
";

%feature("docstring") rfr::trees::tree_options::tree_options "
`tree_options()`  

Default constructor that initializes the values with their default  
";

%feature("docstring") rfr::trees::tree_options::tree_options "
`tree_options(rfr::data_containers::base< num_t, response_t, index_t > &data)`  

Constructor that adjusts the number of features considered at each split
proportional to the square root of the number of features.  
";

%feature("docstring") rfr::trees::tree_options::adjust_limits_to_data "
`adjust_limits_to_data(const rfr::data_containers::base< num_t, response_t,
    index_t > &data)`  
";

%feature("docstring") rfr::trees::tree_options::print_info "
`print_info()`  
";

// File: classrfr_1_1util_1_1weighted__running__statistics.xml


%feature("docstring") rfr::util::weighted_running_statistics "

simple class to compute weighted mean and variance sequentially one value at a
time  

C++ includes: util.hpp
";

%feature("docstring") rfr::util::weighted_running_statistics::weighted_running_statistics "
`weighted_running_statistics()`  
";

%feature("docstring") rfr::util::weighted_running_statistics::weighted_running_statistics "
`weighted_running_statistics(num_t m, num_t s, running_statistics< num_t >
    w_stat)`  
";

%feature("docstring") rfr::util::weighted_running_statistics::serialize "
`serialize(Archive &archive)`  
";

%feature("docstring") rfr::util::weighted_running_statistics::push "
`push(num_t x, num_t weight)`  
";

%feature("docstring") rfr::util::weighted_running_statistics::pop "
`pop(num_t x, num_t weight)`  
";

%feature("docstring") rfr::util::weighted_running_statistics::number_of_points "
`number_of_points() const -> long unsigned int`  

returns the number of points  

Returns
-------
the current number of points added  
";

%feature("docstring") rfr::util::weighted_running_statistics::squared_deviations_from_the_mean "
`squared_deviations_from_the_mean() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::divide_sdm_by "
`divide_sdm_by(num_t fraction, num_t min_weight) const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::mean "
`mean() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::sum_of_weights "
`sum_of_weights() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::sum_of_squares "
`sum_of_squares() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::variance_population "
`variance_population() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::variance_unbiased_frequency "
`variance_unbiased_frequency() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::variance_unbiased_importance "
`variance_unbiased_importance() const -> num_t`  
";

%feature("docstring") rfr::util::weighted_running_statistics::multiply_weights_by "
`multiply_weights_by(const num_t a) const -> weighted_running_statistics`  
";

%feature("docstring") rfr::util::weighted_running_statistics::numerically_equal "
`numerically_equal(weighted_running_statistics other, num_t rel_error) -> bool`  
";

%feature("docstring") rfr::util::weighted_running_statistics::get_weight_statistics "
`get_weight_statistics() const -> running_statistics< num_t >`  
";

// File: namespacerfr.xml

%feature("docstring") rfr::data_containers::read_csv_file "
`read_csv_file(std::string filename) -> std::vector< std::vector< num_type > >`  

A utility function that reads a csv file containing only numerical values.  

A very common use case should be reading the 'training data' from a file in csv
format. This function does that assuming that each row has the same number of
entries. It does NOT read any header information.  

Parameters
----------
* `filename` :  
    the CSV file to be read  

Returns
-------
The data in a 2d 'array' ready to be used by the data container classes  
";

%feature("docstring") rfr::data_containers::print_vector "
`print_vector(const std::vector< T > &v)`  
";

%feature("docstring") rfr::data_containers::print_matrix "
`print_matrix(const std::vector< std::vector< T > > &v)`  
";

// File: namespacerfr_1_1data__containers.xml

// File: namespacerfr_1_1forests.xml

// File: namespacerfr_1_1nodes.xml

// File: namespacerfr_1_1splits.xml

// File: namespacerfr_1_1trees.xml

// File: namespacerfr_1_1util.xml

%feature("docstring") rfr::util::disjunction "
`disjunction(const std::vector< bool > &source, std::vector< bool > &dest)`  
";

%feature("docstring") rfr::util::get_non_NAN_indices "
`get_non_NAN_indices(const std::vector< num_t > &vector) -> std::vector<
    unsigned int >`  
";

%feature("docstring") rfr::util::any_true "
`any_true(const std::vector< bool > &b_vector, const std::vector< unsigned int >
    indices) -> bool`  
";

%feature("docstring") rfr::util::subspace_cardinality "
`subspace_cardinality(const std::vector< std::vector< num_t > > &subspace,
    std::vector< index_t > types) -> num_t`  
";

%feature("docstring") rfr::util::merge_two_vectors "
`merge_two_vectors(num_t *f1, num_t *f2, num_t *dest, index_type n)`  
";

// File: array__wrapper_8hpp.xml

// File: data__container_8hpp.xml

// File: data__container__utils_8hpp.xml

// File: default__data__container_8hpp.xml

// File: default__data__container__with__instances_8hpp.xml

// File: classification__forest_8hpp.xml

// File: fanova__forest_8hpp.xml

// File: forest__options_8hpp.xml

// File: mondrian__forest_8hpp.xml

// File: quantile__regression__forest_8hpp.xml

// File: regression__forest_8hpp.xml

// File: k__ary__mondrian__node_8hpp.xml

// File: k__ary__node_8hpp.xml

// File: temporary__node_8hpp.xml

// File: binary__split__one__feature__rss__loss_8hpp.xml

// File: classification__split_8hpp.xml

// File: split__base_8hpp.xml

// File: binary__fanova__tree_8hpp.xml

// File: k__ary__mondrian__tree_8hpp.xml

// File: k__ary__tree_8hpp.xml

// File: tree__base_8hpp.xml

// File: tree__options_8hpp.xml

// File: util_8hpp.xml

// File: dir_4bbddff8457a34332d257e49dd43fe57.xml

// File: dir_81e33cff5ed926ee1df06d247494dcb0.xml

// File: dir_d44c64559bbebec7f509842c48db8b23.xml

// File: dir_842b872bd45b5b93aec38781b7c6ed18.xml

// File: dir_5df3820c1adf88082d09f24e0d39a432.xml

// File: dir_fc9c8c79b4db954065d202c034a17898.xml

// File: dir_6ccb83c5697ef8790f2ff792e49935de.xml


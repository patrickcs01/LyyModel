class FeatureTools():
    def __init__(self, feature_head_name: list = [],
                 numeric_feature_names: list = [],
                 categorical_features_with_vocabulary: dict = {},
                 target_name: str = "", target_feature_labels: list = [], **kwargs):
        assert len(feature_head_name) != 0 and \
               len(target_feature_labels) != 0

        self.feature_head = feature_head_name
        self.target_feature_labels = target_feature_labels
        self.target_name = target_name
        self.numeric_feature_names = numeric_feature_names
        self.categorical_features_with_vocabulary = categorical_features_with_vocabulary
        self.categorical_features_names = list(categorical_features_with_vocabulary.keys())
        self.all_feature_name = self.categorical_features_names + self.numeric_feature_names
        self.nums_class = len(target_feature_labels)

        self.column_defaults= [
            [0] if feature_name in numeric_feature_names + [target_name] else ["NA"]
            for feature_name in feature_head_name
        ]

import ast
import re
import json

class ConstraintParser:
    def __init__(self, filename=None):
        self.filename = filename
        self.constraints_dict = {}

    @staticmethod
    def parse_condition(condition):
        """Parse a single condition string into a list of dictionaries with feature, operator, and value."""
        parts = re.split(r" (<=|>=|<|>|==) ", condition.strip())
        if len(parts) == 3:
            feature, operator, value = parts
            return [{"feature": feature.strip(), "operator": operator, "value": float(value.strip())}]
        elif len(parts) == 5:
            value1, operator1, feature, operator2, value2 = parts
            return [
                {"feature": feature.strip(), "operator": operator1, "value": float(value1.strip())},
                {"feature": feature.strip(), "operator": operator2, "value": float(value2.strip())}
            ]
        else:
            return None

    @staticmethod
    def constraints_v1_to_dict(raw_string):
        stripped_string = raw_string.replace("Class Bounds: ", "").strip()
        parsed_dict = ast.literal_eval(stripped_string)
        nested_dict = {}
        for class_name, conditions in parsed_dict.items():
            nested_conditions = []
            for condition in conditions:
                parsed_conditions = ConstraintParser.parse_condition(condition)
                if parsed_conditions:
                    nested_conditions.extend(parsed_conditions)
            nested_dict[class_name] = nested_conditions
        return nested_dict

    @staticmethod
    def transform_by_feature(nested_dict):
        feature_dict = {}
        for class_name, conditions in nested_dict.items():
            for condition in conditions:
                feature = condition["feature"]
                if feature not in feature_dict:
                    feature_dict[feature] = []
                feature_dict[feature].append({"class": class_name, "operator": condition["operator"], "value": condition["value"]})
        return feature_dict

    @staticmethod
    def get_intervals_by_feature(feature_based_dict):
        feature_intervals = {}
        for feature, conditions in feature_based_dict.items():
            lower_bound = float('-inf')
            upper_bound = float('inf')
            for condition in conditions:
                operator = condition["operator"]
                value = condition["value"]
                if operator == "<":
                    upper_bound = min(upper_bound, value)
                elif operator == "<=":
                    upper_bound = min(upper_bound, value)
                elif operator == ">":
                    lower_bound = max(lower_bound, value)
                elif operator == ">=":
                    lower_bound = max(lower_bound, value)
            feature_intervals[feature] = (lower_bound, upper_bound)
        return feature_intervals

    @staticmethod
    def is_value_valid_for_class(class_name, feature, value, nested_dict):
        conditions = nested_dict.get(class_name, [])
        for condition in conditions:
            if condition["feature"] == feature:
                operator = condition["operator"]
                comparison_value = condition["value"]
                if operator == "<" and not (value < comparison_value):
                    return False
                elif operator == "<=" and not (value <= comparison_value):
                    return False
                elif operator == ">" and not (value > comparison_value):
                    return False
                elif operator == ">=" and not (value >= comparison_value):
                    return False
        return True

    def read_constraints_from_file(self):
        with open(self.filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                class_label, json_string = line.split(":", 1)
                json_string = json_string.strip().replace("'", '"').replace("None", "null")
                try:
                    self.constraints_dict[class_label.strip()] = json.loads(json_string)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {class_label}: {e}")
        return self.constraints_dict

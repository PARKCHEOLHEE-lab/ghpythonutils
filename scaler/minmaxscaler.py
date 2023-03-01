__all__ = ["MinMaxScaler"]


class MinMaxScaler:
    @staticmethod
    def get_normalized_data(values, range_scale=1):
        min_value = min(values)
        max_value = max(values)
        values_range = max_value - min_value
        
        normalized_values = []
        for value in values:
            normalized_value = (
                (value - min_value) / values_range
            ) * range_scale
            
            normalized_values.append(normalized_value)
        
        return normalized_values

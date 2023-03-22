class MinMaxScaler:
    @staticmethod
    def get_normalized_data(values, range_scale=1.0):
        """Get normalized data by given range scale

        Args:
            values (List[float]): Original data
            range_scale (float, optional): Max scaled value. Defaults to 1.0.

        Returns:
            List[float]: Normalized data
        """

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

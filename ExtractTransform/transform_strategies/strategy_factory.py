from .abstract_strategy import TransformStrategy
from .bird_transform import BirdTransformStrategy
from .mammal_transform import MammalTransformStrategy


class TransformStrategyFactory:
    """
    Factory to create transformation strategies based on the category.

    This class provides a method to obtain the appropriate transformation strategy
    instance for a given category. It supports scalable addition of new categories
    by mapping them to their corresponding strategy classes.
    """

    _strategies = {
        'Bird': lambda: BirdTransformStrategy(),
        'Mammal': lambda: MammalTransformStrategy(),
        # Other mappings here when scaling
    }

    @staticmethod
    def get_strategy(category: str) -> TransformStrategy:
        """
        Retrieve the transformation strategy for the specified category.

        Args:
            category (str): The category for which to retrieve the transformation strategy.

        Returns:
            TransformStrategy: An instance of the transformation strategy corresponding to the category.

        Raises:
            ValueError: If no strategy is defined for the specified category.
        """
        try:
            return TransformStrategyFactory._strategies[category]()
        except KeyError:
            raise ValueError(f"No transformation strategy defined for category: {category}")

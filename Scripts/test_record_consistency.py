import unittest
import pandas as pd

class TestRecordsConsistency(unittest.TestCase):

    def setUp(self):
        # Load the CSV files for testing
        self.records = pd.read_csv('../DATA/records.csv', na_values=[], keep_default_na=False)
        self.species = pd.read_csv('../DATA/species.csv', na_values=[], keep_default_na=False)

        # Merge records with species to include full species information
        self.merged_data = pd.merge(self.records, self.species, on='species_code', how='left')

        # Hard-coded expected values from the original 'birds' DataFrame
        self.expected_data = {
            10476: {'order': 'Anseriformes', 'family': 'Anatidae', 'scientific_name': 'Branta canadensis',
                    'common_name': 'Canada Goose', 'raptor_group': 'N/A'},
            1824: {'order': 'Pelecaniformes', 'family': 'Pelecanidae', 'scientific_name': 'Pelecanus erythrorhynchos',
                   'common_name': 'American White Pelican', 'raptor_group': 'N/A'},
            409: {'order': 'Columbiformes', 'family': 'Columbidae', 'scientific_name': 'Zenaida macroura',
                  'common_name': 'Mourning Dove', 'raptor_group': 'N/A'},
            12149: {'order': 'Strigiformes', 'family': 'Strigidae', 'scientific_name': 'Aegolius acadicus',
                    'common_name': 'Northern Saw-Whet Owl', 'raptor_group': 'Owl'},
            4506: {'order': 'Gruiformes', 'family': 'Rallidae', 'scientific_name': 'Fulica americana',
                   'common_name': 'American Coot', 'raptor_group': 'N/A'},
            4012: {'order': 'Charadriiformes', 'family': 'Scolopacidae', 'scientific_name': 'Calidris bairdii',
                   'common_name': "Baird's Sandpiper", 'raptor_group': 'N/A'},
            3657: {'order': 'Passeriformes', 'family': 'Parulidae', 'scientific_name': 'Vermivora ruficapilla',
                   'common_name': 'Nashville Warbler', 'raptor_group': 'N/A'},
            2286: {'order': 'Pelecaniformes', 'family': 'Ardeidae', 'scientific_name': 'Butorides virescens',
                   'common_name': 'Green Heron', 'raptor_group': 'N/A'},
            12066: {'order': 'Passeriformes', 'family': 'Parulidae', 'scientific_name': 'Dendroica coronata',
                    'common_name': 'Yellow-Rumped Warbler', 'raptor_group': 'N/A'},
            1679: {'order': 'Charadriiformes', 'family': 'Scolopacidae', 'scientific_name': 'Numenius americanus',
                   'common_name': 'Long-Billed Curlew', 'raptor_group': 'N/A'}
        }

    def test_data_reconstruction(self):
        # Verify that each expected index can be correctly reconstructed from merged data
        for index, expected in self.expected_data.items():
            # Retrieve the row from the merged DataFrame corresponding to the index
            merged_row = self.merged_data.iloc[index]

            # Verify each field matches the expected values
            for field in ['order', 'family', 'scientific_name', 'common_name', 'raptor_group']:
                actual_value = merged_row[field]
                expected_value = expected[field]

                # Print debugging info if values mismatch
                if actual_value != expected_value:
                    print(f"Mismatch in {field} for index {index}: expected {expected_value}, got {actual_value}")

                self.assertEqual(actual_value, expected_value,
                                 f"Mismatch in {field} for index {index}: expected {expected_value}, got {actual_value}")

if __name__ == '__main__':
    unittest.main()

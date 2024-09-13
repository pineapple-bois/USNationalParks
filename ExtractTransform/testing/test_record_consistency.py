import unittest
import pandas as pd


class TestRecordsConsistency(unittest.TestCase):

    def setUp(self):
        # Load the CSV files for testing
        self.records = pd.read_pickle('Data/record_master.pkl')
        self.birds = pd.read_pickle('Data/bird_master.pkl')
        self.mammals = pd.read_pickle('Data/mammal_master.pkl')

        # Concatenate birds and mammals into a single species DataFrame
        self.species = pd.concat([self.birds, self.mammals], ignore_index=True)

        # Merge records with the combined species DataFrame on 'species_code'
        self.merged_data = pd.merge(self.records, self.species, on='species_code', how='left')

        # Hard-coded expected values from the original DataFrame (from transform_records.log)
        self.expected_data = {
            15709: {'order': 'Cetacea', 'family': 'Physeteridae', 'scientific_name': 'Physeter macrocephalus',
                    'common_names': 'Sperm Whale'},
            12391: {'order': 'Anseriformes', 'family': 'Anatidae', 'scientific_name': 'Anas discors',
                    'common_names': 'Blue-Winged Teal'},
            1145: {'order': 'Passeriformes', 'family': 'Tyrannidae', 'scientific_name': 'Myiarchus cinerascens',
                   'common_names': 'Ash-Throated Flycatcher'},
            12109: {'order': 'Passeriformes', 'family': 'Tyrannidae', 'scientific_name': 'Pyrocephalus rubinus',
                    'common_names': 'Vermilion Flycatcher'},
            16041: {'order': 'Lagomorpha', 'family': 'Leporidae', 'scientific_name': 'Sylvilagus floridanus',
                    'common_names': 'Eastern Cottontail'},
            7062: {'order': 'Charadriiformes', 'family': 'Scolopacidae', 'scientific_name': 'Tringa melanoleuca',
                   'common_names': 'Greater Yellowlegs'},
            15573: {'order': 'Carnivora', 'family': 'Canidae', 'scientific_name': 'Canis lupus',
                    'common_names': 'Gray Wolf, Wolf'},
            13328: {'order': 'Strigiformes', 'family': 'Strigidae', 'scientific_name': 'Strix varia',
                    'common_names': 'Barred Owl'},
            16018: {'order': 'Carnivora', 'family': 'Mustelidae', 'scientific_name': 'Martes pennanti',
                    'common_names': 'Fisher'},
            9721: {'order': 'Passeriformes', 'family': 'Parulidae', 'scientific_name': 'Vermivora celata',
                   'common_names': 'Orange-Crowned Warbler'},
            13358: {'order': 'Anseriformes', 'family': 'Anatidae', 'scientific_name': 'Bucephala islandica',
                    'common_names': "Barrow's Goldeneye"},
            11208: {'order': 'Anseriformes', 'family': 'Anatidae', 'scientific_name': 'Histrionicus histrionicus',
                    'common_names': 'Harlequin Duck'},
            16621: {'order': 'Carnivora', 'family': 'Mustelidae', 'scientific_name': 'Mustela erminea muricus',
                    'common_names': 'Ermine (muricus subspecies)'},
            11362: {'order': 'Gruiformes', 'family': 'Rallidae', 'scientific_name': 'Fulica americana',
                    'common_names': 'American Coot'},
            12228: {'order': 'Passeriformes', 'family': 'Cardinalidae', 'scientific_name': 'Passerina amoena',
                    'common_names': 'Lazuli Bunting'}
        }

    def test_data_reconstruction(self):
        # Verify that each expected index can be correctly reconstructed from merged data
        for index, expected in self.expected_data.items():
            # Retrieve the row from the merged DataFrame corresponding to the index
            merged_row = self.merged_data.iloc[index]

            # Verify each field matches the expected values
            for field in ['order', 'family', 'scientific_name', 'common_names']:
                actual_value = merged_row[field]
                expected_value = expected[field]

                # Print debugging info if values mismatch
                if actual_value != expected_value:
                    print(f"Mismatch in {field} for index {index}: expected {expected_value}, got {actual_value}")

                self.assertEqual(actual_value, expected_value,
                                 f"Mismatch in {field} for index {index}: expected {expected_value}, got {actual_value}")

if __name__ == '__main__':
    unittest.main()

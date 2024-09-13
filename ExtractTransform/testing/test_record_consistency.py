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
            3648: {'order': 'Passeriformes', 'family': 'Parulidae', 'scientific_name': 'Seiurus aurocapilla',
                   'common_names': 'Ovenbird'},
            819: {'order': 'Cuculiformes', 'family': 'Cuculidae', 'scientific_name': 'Coccyzus erythropthalmus',
                  'common_names': 'Black-Billed Cuckoo'},
            9012: {'order': 'Chiroptera', 'family': 'Vespertilionidae', 'scientific_name': 'Myotis thysanodes',
                   'common_names': 'Fringed Myotis'},
            8024: {'order': 'Rodentia', 'family': 'Heteromyidae', 'scientific_name': 'Perognathus flavescens',
                   'common_names': 'Plains Pocket Mouse'},
            7314: {'order': 'Passeriformes', 'family': 'Parulidae', 'scientific_name': 'Dendroica coronata',
                   'common_names': 'Yellow-Rumped Warbler'},
            4572: {'order': 'Passeriformes', 'family': 'Tyrannidae', 'scientific_name': 'Empidonax virescens',
                   'common_names': 'Acadian Flycatcher'},
            3358: {'order': 'Carnivora', 'family': 'Phocidae', 'scientific_name': 'Mirounga angustirostris',
                   'common_names': 'Northern Elephant Seal'},
            17870: {'order': 'Anseriformes', 'family': 'Anatidae', 'scientific_name': 'Mergus serrator',
                    'common_names': 'Red-Breasted Merganser'},
            2848: {'order': 'Passeriformes', 'family': 'Tyrannidae', 'scientific_name': 'Empidonax oberholseri',
                   'common_names': 'Dusky Flycatcher'},
            13825: {'order': 'Passeriformes', 'family': 'Mimidae', 'scientific_name': 'Toxostoma redivivum',
                    'common_names': 'California Thrasher'},
            1041: {'order': 'Rodentia', 'family': 'Cricetidae', 'scientific_name': 'Neotoma micropus',
                   'common_names': 'Southern Plains Woodrat'},
            976: {'order': 'Strigiformes', 'family': 'Strigidae', 'scientific_name': 'Asio otus',
                  'common_names': 'Long-Eared Owl'},
            3070: {'order': 'Columbiformes', 'family': 'Columbidae', 'scientific_name': 'Streptopelia decaocto',
                   'common_names': 'Eurasian Collared-Dove'},
            7164: {'order': 'Charadriiformes', 'family': 'Scolopacidae', 'scientific_name': 'Actitis macularius',
                   'common_names': 'Spotted Sandpiper'},
            7623: {'order': 'Columbiformes', 'family': 'Columbidae', 'scientific_name': 'Columba livia',
                   'common_names': 'Rock Pigeon'}
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

import pandas as pd
from collections import Counter
from fuzzywuzzy import process, fuzz
from ExtractTransform.utils import DataFrameUtils


class DataFrameTransformation:
    @staticmethod
    def process_scientific_names(df, condition=2):
        """
        Process the scientific names in the birds DataFrame to categorize and count common names
        based on the word count condition (1, 2, or >2 words).

        Args:
        df (pd.DataFrame): The DataFrame containing the 'scientific_name' and 'common_names' columns.
        condition (int): The word count condition (1 for single-word, 2 for two-word, 3 for more than two words).

        Returns:
        dict: Contains lists of scientific names with no common names, single common names, and multiple common names.
        """
        # Step 1: Create sets based on word count in scientific names
        sci_name_set = set(df['scientific_name'])
        single_sci_names = {name for name in sci_name_set if len(name.split()) == 1}
        standard_sci_names = {name for name in sci_name_set if len(name.split()) == 2}
        extended_sci_names = {name for name in sci_name_set if len(name.split()) > 2}

        # Choose the subset to process based on the condition argument
        if condition == 1:
            target_sci_names = single_sci_names
        elif condition == 2:
            target_sci_names = standard_sci_names
        elif condition == 3:
            target_sci_names = extended_sci_names
        else:
            raise ValueError("Condition must be 1 (single-word), 2 (two-word), or 3 (more than two words).")

        # Step 2: Count occurrences of common names for the selected subset of scientific names
        common_name_counts = {}
        for sci_name in target_sci_names:
            # Extract common names for the current scientific name
            common_names = df[df['scientific_name'] == sci_name]['common_names']
            # Count occurrences of each common name
            all_common_names = [names.strip() for names in common_names.dropna()]
            common_name_counts[sci_name] = Counter(all_common_names)

        # Step 3: Classify scientific names based on their common name counts
        no_common_names = []
        multiple_common_names = []
        single_common_names = []

        for sci_name, counts in common_name_counts.items():
            if not counts:
                no_common_names.append(sci_name)
            elif len(counts) > 1:
                multiple_common_names.append((sci_name, counts))
            else:
                single_common_names.append((sci_name, counts))

        # Return results as a dictionary for further use
        return {
            'no_common_names': no_common_names,
            'multiple_common_names': multiple_common_names,
            'single_common_names': single_common_names
        }


    @staticmethod
    def fuzzy_match_scientific_names(sci_names_to_query, df, threshold=90, limit=5):
        """
        Performs fuzzy matching on scientific names that have no associated common names
        to find potential matches in the full dataset, aiming to improve common name associations.

        Args:
            sci_names_to_query (list): List of scientific names without associated common names.
            df (pd.DataFrame): DataFrame containing species records.
            threshold (int): Similarity score threshold for considering matches.
            limit (int): Maximum number of matches to return for each scientific name.

        Returns:
            dict: A dictionary with scientific names as keys and a dictionary of matches
                  and most common names as values, excluding exact matches.
        """
        potential_matches = {}
        all_sci_names = set(df['scientific_name'].unique())

        for sci_name in sci_names_to_query:
            # Check if the scientific name includes subspecies information
            sci_name_parts = sci_name.split()
            if len(sci_name_parts) == 3:
                genus_species = ' '.join(sci_name_parts[:2])  # Extract Genus and species
            else:
                genus_species = sci_name  # Use full name for matching if not a subspecies

            # Use fuzzy matching to find the best matches in the list of all scientific names
            matches = process.extract(genus_species, all_sci_names, scorer=fuzz.ratio, limit=limit)

            # Filter matches to include only those with a similarity score over the threshold
            # and exclude exact matches to the original name
            high_quality_matches = [
                match for match in matches
                if match[1] > threshold and match[0] != sci_name
            ]

            # Proceed only if we have relevant matches
            if high_quality_matches:
                # Retrieve common names for the matching scientific names
                matched_sci_names = [match[0] for match in high_quality_matches]
                common_names = df[df['scientific_name'].isin(matched_sci_names)]['common_names'].dropna()

                # Count occurrences of each common name
                common_names_counter = Counter([name.strip() for names in common_names for name in names.split(',')])
                most_common_name = common_names_counter.most_common(1)

                # If it's a subspecies, append the subspecies info to the common name
                if len(sci_name_parts) == 3:
                    subspecies = sci_name_parts[2]
                    updated_common_name = f"{most_common_name[0][0]} ({subspecies} subspecies)" if most_common_name else 'No common name found'
                else:
                    updated_common_name = most_common_name[0][0] if most_common_name else 'No common name found'

                potential_matches[sci_name] = {
                    'matches': high_quality_matches,
                    'most_common_name': updated_common_name
                }

        return potential_matches


    @staticmethod
    def standardize_common_names(multiple_common_names, show_ties=False):
        """
        Standardizes the common names for scientific names with multiple associated names,
        prioritizing specific, singular names over compound or list-like names.

        Args:
        multiple_common_names (list): List of tuples containing scientific names and their Counter object with common name counts.

        Returns:
        dict: Mapping of scientific names to their standardized common names, with flagged ties.
        """
        standardized_names = {}
        tie_count = 0  # Initialize tie_count at zero

        for sci_name, counts in multiple_common_names:
            most_common = counts.most_common()
            highest_count = most_common[0][1]
            tied_names = [name for name, count in most_common if count == highest_count]

            if len(tied_names) > 1:
                tie_count += 1
                # Apply rules: Prioritize specific names with "'s" over others and then fewest additional words
                chosen_name = min(
                    tied_names,
                    key=lambda x: (
                        x.count(',') + x.count('/') + len(x.split()),  # Penalize commas, slashes, and extra words
                        0 if "'s" in x else 1,  # Prioritize names with "'s" by giving them a lower score
                        len(x),  # Shorter names preferred
                        x  # Alphabetical order as a final fallback
                    )
                )
                if show_ties:
                    print(f"Tie detected for '{sci_name}': {tied_names} with count {highest_count}")
            else:
                chosen_name = tied_names[0]

            standardized_names[sci_name] = chosen_name
        if show_ties:
            print(f"Total ties: {tie_count}")

        return standardized_names


    @staticmethod
    def standardize_common_names_subspecies(multiple_common_names):
        """
        Standardizes common names for scientific names with multiple associated names.
        Prioritizes names with parentheses (indicating subspecies) unless "Race" is in the parentheses.

        Args:
        multiple_common_names (list): List of tuples containing scientific names and their Counter object with common name counts.

        Returns:
        dict: Mapping of scientific names to their standardized common names, with flagged ties.
        """
        standardized_names = {}
        tie_count = 0

        for sci_name, counts in multiple_common_names:
            most_common = counts.most_common()
            names_with_parentheses = [name for name, count in most_common if '(' in name and ')' in name]
            names_with_subspecies_info = [name for name in names_with_parentheses if 'Race' not in name]

            # If names with subspecies information exist, prioritize those
            if names_with_subspecies_info:
                # Choose the most descriptive subspecies name
                chosen_name = max(
                    names_with_subspecies_info,
                    key=lambda x: (x.count('('), len(x.split()), x.count(',') + x.count('/'), len(x))
                )
            elif names_with_parentheses:
                # If only 'Race' or other parentheses exist, still prioritize but less strictly
                chosen_name = max(
                    names_with_parentheses,
                    key=lambda x: (x.count('('), len(x.split()), x.count(',') + x.count('/'), len(x))
                )
            else:
                # fall back to the highest count
                highest_count = most_common[0][1]
                tied_names = [name for name, count in most_common if count == highest_count]

                if len(tied_names) > 1:
                    tie_count += 1
                    chosen_name = min(
                        tied_names,
                        key=lambda x: (x.count(',') + x.count('/') + len(x.split()), len(x), x)
                    )
                else:
                    chosen_name = tied_names[0]

            # Check if the scientific name includes a subspecies or hybrid (Genus species X species) designation
            sci_name_parts = sci_name.split()
            if len(sci_name_parts) > 2:
                # Append the entire remaining string as subspecies designation
                subspecies = ' '.join(sci_name_parts[2:])

                # Wrap chosen name in the form "Genus species ({subspecies} subspecies)" if not already in parentheses
                if '(' not in chosen_name or ')' not in chosen_name:
                    chosen_name = f"{chosen_name} ({subspecies} subspecies)"
            standardized_names[sci_name] = chosen_name

        return standardized_names


    @staticmethod
    def identify_records(dataframe, condition, column_name, logger, category):
        """
        Identifies records based on a condition and logs the results.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.
            condition (int): The condition type to filter records.
            column_name (str): Column name to identify the records.
            logger: Logger object for logging.
            category (str): The category of the records (e.g., 'Bird', 'Mammal').

        Returns:
            tuple: A DataFrame of identified records and the list of identified names.
        """
        results = DataFrameTransformation.process_scientific_names(dataframe, condition=condition)
        identified_names = results[column_name]
        identified_records = dataframe[dataframe['scientific_name'].isin(identified_names)]
        logger.info(f"Found {len(identified_records)} records for condition {condition}.")

        # Only log saving if the dataframe isn't empty
        if not identified_records.empty:
            DataFrameUtils.save_dataframe_to_csv(
                identified_records,
                f"Pipeline/BackupData/{category}",
                f"{column_name}_records.csv",
                logger
            )

        return identified_records, identified_names


    @staticmethod
    def fuzzy_match_and_update(dataframe, sci_names, update_field, logger, category, identifier=None):
        """
        Performs fuzzy matching and updates the DataFrame based on matches.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.
            sci_names (list): List of scientific names to match.
            update_field (str): Field to update ('common_names' or 'scientific_name').
            logger: Logger object for logging.
            category (str): The category of the records (e.g., 'Bird', 'Mammal').
            identifier (any): Postfix value for filename

        Returns:
            pd.DataFrame: Updated DataFrame.
        """
        matches = DataFrameTransformation.fuzzy_match_scientific_names(sci_names, dataframe)
        updated_names = set()

        for sci_name, info in matches.items():
            most_common_name = info['most_common_name']
            if most_common_name == 'No common name found':
                logger.info(f"Skipping update for {sci_name} as no valid common name was found.")
                continue  # Skip updating if no valid common name is found

            best_match = info['matches'][0][0]  # Get the best match
            dataframe.loc[dataframe['scientific_name'] == sci_name, update_field] = most_common_name
            updated_names.add(sci_name)

        logger.info(f"Matched & updated {len(matches)} common_names")

        # Drop records that were not updated
        records_to_drop = dataframe[
            dataframe['scientific_name'].isin(sci_names) & ~dataframe['scientific_name'].isin(updated_names)
            ]
        if not records_to_drop.empty:
            logger.info(f"Dropping {len(records_to_drop)} records that were not updated.")
            filename = f"unmatched_records_{identifier}.csv"  # Unique filename
            DataFrameUtils.save_dataframe_to_csv(
                records_to_drop,
                f"Pipeline/BackupData/{category}",
                filename,
                logger
            )
        return dataframe.drop(records_to_drop.index)


    @staticmethod
    def standardize_names(dataframe, multiple_common_names, standardize_method, logger, category, condition):
        """
        Standardizes common names for scientific names with multiple common names.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.
            multiple_common_names (list): List of tuples with scientific names and their common names counts.
            standardize_method (function): Function to standardize names.
            logger: Logger object for logging.
            category (str): The category of the records (e.g., 'Bird', 'Mammal').

        Returns:
            pd.DataFrame: Updated DataFrame.
        """
        standardized_name_mapping = standardize_method(multiple_common_names)
        logger.info(f"Standardizing {len(standardized_name_mapping)} scientific names with multiple common names.")

        multi_sci_names_list = [sci_name for sci_name, counts in multiple_common_names]
        multi_common_names_records = dataframe[dataframe['scientific_name'].isin(multi_sci_names_list)]
        if not multi_common_names_records.empty:
            DataFrameUtils.save_dataframe_to_csv(
                multi_common_names_records,
                f"Pipeline/BackupData/{category}",
                "multi_common_names_ambiguities.csv",
                logger
            )

        updated_sci_names = set()
        for sci_name, common_name in standardized_name_mapping.items():
            dataframe.loc[dataframe['scientific_name'] == sci_name, 'common_names'] = common_name
            updated_sci_names.add(sci_name)

        # Conditional dropping: apply if condition == 3 (subspecies)
        if condition == 3:
            # Identify and drop records that were not updated
            records_to_drop = dataframe[
                dataframe['scientific_name'].isin(multi_sci_names_list) & ~dataframe['scientific_name'].isin(
                    updated_sci_names)
                ]

            if not records_to_drop.empty:
                drop_count = len(records_to_drop)
                logger.info(f"Dropping {drop_count} subspecies records that were not updated.")
                filename = f"unstandardized_subspecies_records.csv"
                DataFrameUtils.save_dataframe_to_csv(
                    records_to_drop,
                    f"Pipeline/BackupData/{category}",
                    filename,
                    logger
                )
                dataframe = dataframe.drop(records_to_drop.index)

        return dataframe


    @staticmethod
    def process_comma_separated_names(dataframe, all_sci_names, match_score=90):
        """
        Processes scientific names with comma-separated common names to identify potential typos or ambiguous names
        and returns mappings for scientific and common names.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.
            all_sci_names (list): List of all scientific names for fuzzy matching.
            match_score (int): Threshold for fuzzy match score to consider a match.

        Returns:
            tuple: Two dictionaries mapping scientific names to standardized common names
                   and common names to the most appropriate scientific name.
        """
        two_word_sci_names = dataframe[dataframe['scientific_name'].str.split().str.len() == 2]
        comma_separated_common_names = two_word_sci_names[
            two_word_sci_names['common_names'].str.contains(',', na=False)]
        subset_sci_names = comma_separated_common_names['scientific_name'].unique()
        potential_matches = {}

        for sci_name in subset_sci_names:
            genus_species = ' '.join(sci_name.split()[:2])
            matches = process.extract(genus_species, all_sci_names, scorer=fuzz.ratio, limit=5)
            high_quality_matches = [match for match in matches if match[1] > match_score]

            if high_quality_matches:
                matched_sci_names = [match[0] for match in high_quality_matches]
                common_names = dataframe[dataframe['scientific_name'].isin(matched_sci_names)]['common_names'].dropna()
                common_names_counter = Counter(common_names)

                if len(common_names_counter) > 1:
                    most_common_name = common_names_counter.most_common(1)
                    potential_matches[sci_name] = {
                        'matches': high_quality_matches,
                        'most_common_name': most_common_name[0][0] if most_common_name else 'No common name found',
                        'common_name_counts': common_names_counter
                    }

        return potential_matches


    @staticmethod
    def identify_sci_name_ambiguities(dataframe, logger, category):
        """Identify potential typos or ambiguities in common names linked to 'Genus species' scientific names."""
        common_name_to_sci_names = {}
        two_word_sci_names = dataframe[dataframe['scientific_name'].str.split().apply(len) == 2]

        for common_name in two_word_sci_names['common_names'].unique():
            associated_sci_names = two_word_sci_names[two_word_sci_names['common_names'] == common_name][
                'scientific_name']
            sci_name_counts = Counter(associated_sci_names)
            common_name_to_sci_names[common_name] = sci_name_counts

        potential_issues = {
            common_name: sci_name_counts
            for common_name, sci_name_counts in common_name_to_sci_names.items()
            if len(sci_name_counts) > 1
        }

        potential_issues_df = pd.DataFrame([
            {'common_names': common_name,
             'scientific_name': sci_name,
             'record_count': count
             }
            for common_name, sci_name_counts in potential_issues.items()
            for sci_name, count in sci_name_counts.items()
        ])
        if not potential_issues_df.empty:
            logger.info(f"Identified {len(potential_issues_df)} potential ambiguities in common names.")
            DataFrameUtils.save_dataframe_to_csv(
                potential_issues_df,
                f"Pipeline/BackupData/{category}",
                "scientific_name_ambiguities.csv",
                logger
            )
        else:
            logger.info("No ambiguities found in common names.")

        return potential_issues_df


    @staticmethod
    def select_scientific_names_by_common_name(potential_issues_df):
        """Selects scientific names based on highest count and flags ties for review."""
        selected_sci_names = {}
        ties_for_review = {}

        for common_name, group in potential_issues_df.groupby('common_names'):
            max_count = group['record_count'].max()
            max_count_names = group[group['record_count'] == max_count]

            if len(max_count_names) > 1:
                ties_for_review[common_name] = max_count_names[['scientific_name', 'record_count']].to_dict(
                    orient='records')
            else:
                chosen_name = max_count_names.iloc[0]['scientific_name']
                selected_sci_names[common_name] = chosen_name

        return selected_sci_names, ties_for_review


    @staticmethod
    def update_scientific_names(dataframe, selected_sci_names):
        """Updates the scientific names in the birds DataFrame based on selected mappings."""
        selected_sci_names_mapping = {common_name: sci_name for common_name, sci_name in selected_sci_names.items()}

        dataframe['scientific_name'] = dataframe.apply(
            lambda row: selected_sci_names_mapping.get(row['common_names'], row['scientific_name'])
            if len(row['scientific_name'].split()) == 2 else row['scientific_name'],
            axis=1
        )
        return dataframe


    @staticmethod
    def identify_subspecies(dataframe):
        """
        Identifies records in the DataFrame with scientific names of the form 'Genus species subspecies'.
        """
        three_word_sci_names = dataframe[dataframe['scientific_name'].str.split().str.len() >= 3].copy()
        three_word_sci_names['genus_species'] = three_word_sci_names['scientific_name'].apply(
                                    lambda x: ' '.join(x.split()[:2]))
        return three_word_sci_names


    @staticmethod
    def map_genus_species_to_common_names(full_dataframe, subspecies_df):
        """
        Maps genus_species to common names using the genus-species records from the full DataFrame.
        """
        two_word_sci_names = full_dataframe[full_dataframe['scientific_name'].str.split().str.len() == 2]
        genus_species_to_common_name = two_word_sci_names.set_index('scientific_name')['common_names'].to_dict()
        subspecies_df['matched_common_name'] = subspecies_df['genus_species'].map(genus_species_to_common_name)
        return subspecies_df


    @staticmethod
    def standardize_common_names_helper(row):
        """
        Standardizes the common name for a subspecies row by appending the subspecies information.
        """
        subspecies = row['scientific_name'].split()[-1]
        if '(' not in row['common_names']:
            return f"{row['matched_common_name']} ({subspecies} subspecies)"
        return row['common_names']


    @staticmethod
    def filter_and_standardize_subspecies_names(subspecies_df, category, config, logger):
        """
        Filters and standardizes subspecies common names based on configuration from a YAML file.
        """
        # Load exclusions and columns to drop from the config
        common_names_to_exclude = config.get('common_names_to_exclude', {}).get(category, [])
        columns_to_exclude = config.get('columns_to_exclude', {}).get('default', [])

        # Check if there are exclusions for the current category
        if not common_names_to_exclude:
            logger.info(
                f"No specific common names to exclude found for category '{category}'. Proceeding without exclusions.")
        else:
            subspecies_df = subspecies_df[~subspecies_df['matched_common_name'].isin(common_names_to_exclude)]

        # Check if there are columns to exclude from the DataFrame
        if not columns_to_exclude:
            logger.info(f"No specific columns to exclude found. Proceeding without column exclusions.")
        else:
            subspecies_df = subspecies_df.drop(columns=columns_to_exclude, errors='ignore')

        # Drop rows with missing matched common names
        subspecies_df = subspecies_df.dropna(subset=['matched_common_name'])

        # Apply standardization of common names
        if subspecies_df.empty:
            logger.info(f"No subspecies records to standardize for category '{category}'.")
        else:
            subspecies_df['common_names'] = subspecies_df.apply(
                DataFrameTransformation.standardize_common_names_helper, axis=1
            )
        return subspecies_df

import pandas as pd
from collections import Counter


def process_scientific_names(df, condition=2):
    """
    Process the scientific names in the birds DataFrame to categorize and count common names
    based on the word count condition (1, 2, or >2 words).

    Args:
    birds_df (pd.DataFrame): The DataFrame containing the 'scientific_name' and 'common_names' columns.
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
        print("Single Scientific Name Count (1 word):", len(single_sci_names))
    elif condition == 2:
        target_sci_names = standard_sci_names
        print("Standard Scientific Names Count (2 words):", len(standard_sci_names))
    elif condition == 3:
        target_sci_names = extended_sci_names
        print("Extended Scientific Names Count (> 2 words):", len(extended_sci_names))
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

    # Display results
    print(f"Scientific names with no associated common names: {len(no_common_names)}")
    print(f"Scientific names with multiple associated common names: {len(multiple_common_names)}")
    print(f"Scientific names with a single associated common name: {len(single_common_names)}")

    # Return results as a dictionary for further use
    return {
        'no_common_names': no_common_names,
        'multiple_common_names': multiple_common_names,
        'single_common_names': single_common_names
    }


def standardize_common_names(multiple_common_names):
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
        # Get the most common names and their counts
        most_common = counts.most_common()

        # Get the highest count
        highest_count = most_common[0][1]
        tied_names = [name for name, count in most_common if count == highest_count]

        if len(tied_names) > 1:
            tie_count += 1
            # print(f"Tie detected for '{sci_name}': {tied_names} with count {highest_count}")

            # Apply rules: Prioritize specific names without commas or multiple options
            # Rule: Choose names with the fewest commas or additional words
            chosen_name = min(
                tied_names,
                key=lambda x: (x.count(',') + x.count('/') + len(x.split()), len(x), x)
            )
        else:
            # If no tie, use the most common name
            chosen_name = tied_names[0]

        # Map the scientific name to the chosen common name
        standardized_names[sci_name] = chosen_name

    # Print the total number of ties detected
    print(f"Total ties: {tie_count}")

    return standardized_names


def standardize_common_names_subspecies(multiple_common_names, show_ties=False):
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
            # If no parentheses, fall back to the highest count
            highest_count = most_common[0][1]
            tied_names = [name for name, count in most_common if count == highest_count]

            if len(tied_names) > 1:
                tie_count += 1
                # Apply the rule: Choose the simplest and most descriptive name
                chosen_name = min(
                    tied_names,
                    key=lambda x: (x.count(',') + x.count('/') + len(x.split()), len(x), x)
                )
                if show_ties:
                    print(f"Tie detected for '{sci_name}': {tied_names} with count {highest_count}")
            else:
                chosen_name = tied_names[0]

        standardized_names[sci_name] = chosen_name
    print(f"Total ties: {tie_count}")
    return standardized_names

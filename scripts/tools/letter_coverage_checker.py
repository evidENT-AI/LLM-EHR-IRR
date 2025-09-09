#!/usr/bin/env python3

"""
Letter Assignment Analysis Tool

This script analyzes the distribution of letter assignments among annotators
in a research project. It identifies various issues in the assignment system:

- Duplicate letter assignments within individual annotators
- Missing letter assignments (letters with no annotators)
- Under-assigned letters (letters with only 1 annotator instead of 2)
- Over-assigned letters (letters with 3+ annotators instead of 2)
- Unexpected letter assignments (letters outside the expected range)
- Workload distribution across all annotators

The script expects a JSON input file containing user data with assigned letters.
Multiple JSON formats are supported:
- A list of user objects
- A dictionary with a 'users' key containing user objects
- A dictionary where each key contains a user object with 'username' field

Usage:
    python letter_assignment_analysis.py [input_file]
    
Returns:
    A dictionary containing detailed analysis results for potential fixing of issues
    
Example:
    python letter_assignment_analysis.py users.json
"""

import json
from collections import defaultdict

def analyze_letter_assignments(input_file):
    """
    Analyze the letter assignments in the user data to identify issues:
    1. Check for duplicate letter assignments within users
    2. Verify each letter (0000-0098, excluding 0080) has exactly 2 annotators
    3. Analyze workload distribution across annotators
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Convert to list of users if necessary
    if isinstance(data, dict) and 'users' in data:
        users = list(data['users'].values())
    elif isinstance(data, list):
        users = data
    else:
        # Try to parse as individual JSON objects
        users = []
        for key, value in data.items():
            if isinstance(value, dict) and 'username' in value:
                users.append(value)

    print(f"Analyzing assignments for {len(users)} users")

    # Initialize tracking structures
    letter_to_annotators = defaultdict(list)
    annotator_to_letters = {}
    annotator_duplicates = {}

    # Expected letters (0000-0098, excluding 0080)
    expected_letters = {f"{i:04d}" for i in range(99) if i != 80}

    # Process each user's assigned letters
    for user in users:
        username = user.get('username')
        assigned_letters = user.get('assignedLetters', [])

        # Check for duplicates within this user's assignments
        unique_letters = set()
        duplicates = []

        for letter in assigned_letters:
            if letter in unique_letters:
                duplicates.append(letter)
            unique_letters.add(letter)

        if duplicates:
            annotator_duplicates[username] = duplicates

        # Store unique letter assignments
        annotator_to_letters[username] = list(unique_letters)

        # Update letter -> annotators mapping
        for letter in unique_letters:
            letter_to_annotators[letter].append(username)

    # Analyze coverage
    missing_letters = expected_letters - set(letter_to_annotators.keys())

    under_assigned = []
    correctly_assigned = []
    over_assigned = []

    for letter, annotators in letter_to_annotators.items():
        if letter in expected_letters:
            if len(annotators) < 2:
                under_assigned.append(letter)
            elif len(annotators) == 2:
                correctly_assigned.append(letter)
            else:
                over_assigned.append(letter)

    # Check for unexpected letters
    unexpected_letters = set(letter_to_annotators.keys()) - expected_letters

    # Analyze workload distribution
    workloads = [(username, len(letters)) for username, letters in annotator_to_letters.items()]
    workloads.sort(key=lambda x: x[1])

    min_load = workloads[0][1] if workloads else 0
    max_load = workloads[-1][1] if workloads else 0
    avg_load = sum(load for _, load in workloads) / len(workloads) if workloads else 0
    
    # Print analysis results
    print("\n===== LETTER ASSIGNMENT ANALYSIS =====")
    print(f"Total users: {len(users)}")
    print(f"Expected letters: {len(expected_letters)}")

    print("\n----- COVERAGE SUMMARY -----")
    print(f"Missing letters: {len(missing_letters)} ({', '.join(sorted(missing_letters)) if missing_letters else 'None'})")
    print(f"Under-assigned letters (1 annotator): {len(under_assigned)} ({', '.join(sorted(under_assigned)) if under_assigned else 'None'})")
    print(f"Correctly assigned letters (2 annotators): {len(correctly_assigned)}")
    print(f"Over-assigned letters (3+ annotators): {len(over_assigned)} ({', '.join(sorted(over_assigned)) if over_assigned else 'None'})")
    print(f"Unexpected letters: {len(unexpected_letters)} ({', '.join(sorted(unexpected_letters)) if unexpected_letters else 'None'})")

    print("\n----- WORKLOAD DISTRIBUTION -----")
    print(f"Min letters per user: {min_load}")
    print(f"Max letters per user: {max_load}")
    print(f"Avg letters per user: {avg_load:.1f}")

    print("\n----- DUPLICATE ASSIGNMENTS -----")
    if annotator_duplicates:
        print(f"{len(annotator_duplicates)} users have duplicate letter assignments:")
        for username, dupes in annotator_duplicates.items():
            print(f"  {username}: {', '.join(dupes)}")
    else:
        print("No duplicate assignments found")

    print("\n----- DETAILED LETTER ASSIGNMENTS -----")
    print("Format: Letter: [annotator1, annotator2, ...]")
    all_letters = sorted(letter_to_annotators.keys())
    for letter in all_letters:
        annotators = letter_to_annotators[letter]
        coverage_status = "✓" if letter in expected_letters and len(annotators) == 2 else "❌"
        print(f"{letter} {coverage_status}: {annotators}")

    # Return complete analysis results for potential use in fixing
    return {
        'letter_to_annotators': dict(letter_to_annotators),
        'annotator_to_letters': annotator_to_letters,
        'missing_letters': missing_letters,
        'under_assigned': under_assigned,
        'over_assigned': over_assigned,
        'unexpected_letters': unexpected_letters,
        'duplicates': annotator_duplicates,
        'workloads': workloads
    }

def fix_letter_assignments(input_file, output_file, analysis=None):
    """
    Fix the letter assignments based on analysis results or by performing a new analysis.
    Writes the fixed assignments to the output file.
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Determine the structure of the data
    if isinstance(data, dict) and 'users' in data:
        users_dict = data['users']
        users = list(users_dict.values())
        output_format = 'dict'
    elif isinstance(data, list):
        users = data
        output_format = 'list'
    else:
        # Try to parse as individual JSON objects
        users_dict = {}
        for key, value in data.items():
            if isinstance(value, dict) and 'username' in value:
                users_dict[key] = value
        users = list(users_dict.values())
        output_format = 'dict'

    # Run analysis if not provided
    if analysis is None:
        # We'll run a simplified version here
        letter_to_annotators = defaultdict(list)
        annotator_to_letters = {}

        # First clean up duplicates
        for user in users:
            username = user['username']
            assigned_letters = user['assignedLetters']

            # Remove duplicates while preserving order
            unique_letters = []
            seen = set()
            for letter in assigned_letters:
                if letter not in seen:
                    unique_letters.append(letter)
                    seen.add(letter)

            # Update user's assigned letters
            user['assignedLetters'] = unique_letters
            annotator_to_letters[username] = unique_letters

            # Update letter -> annotators mapping
            for letter in unique_letters:
                letter_to_annotators[letter].append(username)

        # Expected letters (0000-0098, excluding 80)
        expected_letters = {f"{i:04d}" for i in range(99) if i != 80}

        # Check coverage
        missing_letters = expected_letters - set(letter_to_annotators.keys())
        under_assigned = [l for l, annotators in letter_to_annotators.items() 
                          if l in expected_letters and len(annotators) < 2]
        unexpected_letters = set(letter_to_annotators.keys()) - expected_letters
    else:
        # Use the provided analysis
        letter_to_annotators = defaultdict(list, analysis['letter_to_annotators'])
        annotator_to_letters = analysis['annotator_to_letters']
        missing_letters = analysis['missing_letters']
        under_assigned = analysis['under_assigned']
        unexpected_letters = analysis['unexpected_letters']

        # First clean up duplicates
        for user in users:
            username = user['username']
            if username in annotator_to_letters:
                user['assignedLetters'] = annotator_to_letters[username]

    # Now fix the coverage issues

    # 1. Remove unexpected letters (like 80)
    for user in users:
        username = user['username']
        user['assignedLetters'] = [l for l in user['assignedLetters'] if l not in unexpected_letters]
        annotator_to_letters[username] = [l for l in annotator_to_letters[username] if l not in unexpected_letters]

    # 2. Assign missing and under-assigned letters
    # Get sorted list of users by workload
    workloads = [(username, len(letters)) for username, letters in annotator_to_letters.items()]
    workloads.sort(key=lambda x: x[1])

    # Process missing letters first
    for letter in missing_letters:
        # Assign to the two users with lightest workload
        for i in range(min(2, len(workloads))):
            username = workloads[i][0]
            user = next(u for u in users if u['username'] == username)
            user['assignedLetters'].append(letter)
            annotator_to_letters[username].append(letter)
            letter_to_annotators[letter].append(username)

            # Update workload
            workloads[i] = (username, workloads[i][1] + 1)
            workloads.sort(key=lambda x: x[1])

    # Then process under-assigned letters
    for letter in under_assigned:
        # Find the current annotator
        current_annotators = letter_to_annotators[letter]

        # Find eligible user with lightest workload
        for username, _ in workloads:
            if username not in current_annotators:
                user = next(u for u in users if u['username'] == username)
                user['assignedLetters'].append(letter)
                annotator_to_letters[username].append(letter)
                letter_to_annotators[letter].append(username)

                # Update workload
                workloads = [(u, len(annotator_to_letters[u])) for u, _ in workloads]
                workloads.sort(key=lambda x: x[1])
                break

    # Prepare the output data structure
    if output_format == 'dict':
        # Update the users dictionary
        if isinstance(data, dict) and 'users' in data:
            for user in users:
                data['users'][user['username']] = user
        else:
            data = {user['username']: user for user in users}
    else:
        # Use the list format
        data = users

    # Write the fixed data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nFixed assignments written to {output_file}")

    # Run final verification
    print("\n===== VERIFICATION AFTER FIXES =====")
    current_assignments = {}
    for user in users:
        for letter in user['assignedLetters']:
            if letter not in current_assignments:
                current_assignments[letter] = []
            current_assignments[letter].append(user['username'])

    expected_letters = {f"{i:04d}" for i in range(99) if i != 80}

    # Check if all expected letters have exactly 2 annotators
    all_correct = True
    for letter in expected_letters:
        annotators = current_assignments.get(letter, [])
        if len(annotators) != 2:
            print(f"❌ Letter {letter} has {len(annotators)} annotators: {annotators}")
            all_correct = False

    if all_correct:
        print("✓ All letters have exactly 2 annotators")

    return data

def compare_assignments(original_file, fixed_file):
    """
    Compare the original and fixed letter assignments and print a human-readable diff.
    """
    # Load the original and fixed JSON data
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    with open(fixed_file, 'r') as f:
        fixed_data = json.load(f)
    
    # Extract users from both files
    def extract_users(data):
        if isinstance(data, dict) and 'users' in data:
            return data['users']
        elif isinstance(data, list):
            return {user['username']: user for user in data}
        else:
            return data
    
    original_users = extract_users(original_data)
    fixed_users = extract_users(fixed_data)
    
    # Get all usernames
    all_usernames = set(original_users.keys()) | set(fixed_users.keys()) if isinstance(original_users, dict) else {u['username'] for u in original_users} | {u['username'] for u in fixed_users}
    
    # Compare each user's assignments
    print("\n===== ASSIGNMENT DIFFERENCES =====")
    
    no_differences = True
    for username in sorted(all_usernames):
        # Get assignments for this user
        original_assignments = []
        fixed_assignments = []
        
        if isinstance(original_users, dict):
            if username in original_users:
                original_assignments = original_users[username].get('assignedLetters', [])
            if username in fixed_users:
                fixed_assignments = fixed_users[username].get('assignedLetters', [])
        else:
            original_user = next((u for u in original_users if u['username'] == username), None)
            fixed_user = next((u for u in fixed_users if u['username'] == username), None)
            
            if original_user:
                original_assignments = original_user.get('assignedLetters', [])
            if fixed_user:
                fixed_assignments = fixed_user.get('assignedLetters', [])
        
        # Convert to sets for comparison
        original_set = set(original_assignments)
        fixed_set = set(fixed_assignments)
        
        # Find differences
        removed = original_set - fixed_set
        added = fixed_set - original_set
        
        if removed or added:
            no_differences = False
            print(f"\n{username}:")
            
            if removed:
                print(f"  Removed: {', '.join(sorted(removed))}")
            
            if added:
                print(f"  Added: {', '.join(sorted(added))}")
            
            # Show before and after
            print(f"  BEFORE: {len(original_assignments)} letters - {', '.join(sorted(original_assignments))}")
            print(f"  AFTER:  {len(fixed_assignments)} letters - {', '.join(sorted(fixed_assignments))}")
    
    if no_differences:
        print("No differences found between original and fixed assignments")
    
    # Print letter coverage changes
    print("\n===== LETTER COVERAGE CHANGES =====")
    
    # Build letter coverage maps
    def build_coverage(users_data):
        coverage = defaultdict(list)
        if isinstance(users_data, dict):
            for username, user in users_data.items():
                for letter in user.get('assignedLetters', []):
                    coverage[letter].append(username)
        else:
            for user in users_data:
                username = user['username']
                for letter in user.get('assignedLetters', []):
                    coverage[letter].append(username)
        return coverage
    
    original_coverage = build_coverage(original_users)
    fixed_coverage = build_coverage(fixed_users)
    
    # Find all letters
    all_letters = set(original_coverage.keys()) | set(fixed_coverage.keys())
    
    # Expected letters (0000-0098, excluding 80)
    expected_letters = {f"{i:04d}" for i in range(99) if i != 80}
    
    changes_found = False
    for letter in sorted(all_letters):
        original_annotators = original_coverage.get(letter, [])
        fixed_annotators = fixed_coverage.get(letter, [])
        
        # Check if coverage changed
        if set(original_annotators) != set(fixed_annotators) or len(original_annotators) != len(fixed_annotators):
            changes_found = True
            
            # Determine status text
            status_original = ""
            if letter not in expected_letters:
                status_original = "UNEXPECTED"
            elif len(original_annotators) < 2:
                status_original = "UNDER-ASSIGNED"
            elif len(original_annotators) > 2:
                status_original = "OVER-ASSIGNED"
            
            status_fixed = "FIXED" if len(fixed_annotators) == 2 else "ISSUE REMAINS"
            
            print(f"Letter {letter} ({status_original} → {status_fixed}):")
            print(f"  BEFORE: {len(original_annotators)} annotators - {', '.join(sorted(original_annotators))}")
            print(f"  AFTER:  {len(fixed_annotators)} annotators - {', '.join(sorted(fixed_annotators))}")

    if not changes_found:
        print("No letter coverage changes found")

    # Print summary statistics
    print("\n===== SUMMARY =====")
    
    # Count coverage in original and fixed
    def count_coverage(coverage_map):
        missing = []
        under = []
        correct = []
        over = []
        unexpected = []
        
        for letter, annotators in coverage_map.items():
            if letter not in expected_letters:
                unexpected.append(letter)
            elif len(annotators) == 0:
                missing.append(letter)
            elif len(annotators) < 2:
                under.append(letter)
            elif len(annotators) == 2:
                correct.append(letter)
            else:
                over.append(letter)
        
        # Check for expected letters not in the map
        for letter in expected_letters:
            if letter not in coverage_map:
                missing.append(letter)
        
        return {
            'missing': missing,
            'under_assigned': under,
            'correct': correct,
            'over_assigned': over,
            'unexpected': unexpected
        }
    
    original_stats = count_coverage(original_coverage)
    fixed_stats = count_coverage(fixed_coverage)
    
    print("BEFORE:")
    print(f"  Missing letters: {len(original_stats['missing'])}")
    print(f"  Under-assigned letters: {len(original_stats['under_assigned'])}")
    print(f"  Correctly assigned letters: {len(original_stats['correct'])}")
    print(f"  Over-assigned letters: {len(original_stats['over_assigned'])}")
    print(f"  Unexpected letters: {len(original_stats['unexpected'])}")
    
    print("AFTER:")
    print(f"  Missing letters: {len(fixed_stats['missing'])}")
    print(f"  Under-assigned letters: {len(fixed_stats['under_assigned'])}")
    print(f"  Correctly assigned letters: {len(fixed_stats['correct'])}")
    print(f"  Over-assigned letters: {len(fixed_stats['over_assigned'])}")
    print(f"  Unexpected letters: {len(fixed_stats['unexpected'])}")
    
    # Check for duplicate assignments in original and fixed
    def count_duplicates(users_data):
        duplicates = 0
        if isinstance(users_data, dict):
            for username, user in users_data.items():
                assigned = user.get('assignedLetters', [])
                if len(assigned) != len(set(assigned)):
                    duplicates += 1
        else:
            for user in users_data:
                assigned = user.get('assignedLetters', [])
                if len(assigned) != len(set(assigned)):
                    duplicates += 1
        return duplicates
    
    original_duplicates = count_duplicates(original_users)
    fixed_duplicates = count_duplicates(fixed_users)
    
    print(f"Users with duplicate assignments: {original_duplicates} → {fixed_duplicates}")

    # Print detailed fixed letter assignments
    print("\n===== FIXED LETTER ASSIGNMENTS =====")
    print("Format: Letter: [annotator1, annotator2, ...]")
    all_letters = sorted(fixed_coverage.keys())
    for letter in all_letters:
        annotators = fixed_coverage[letter]
        coverage_status = "✓" if letter in expected_letters and len(annotators) == 2 else "❌"
        print(f"{letter} {coverage_status}: {', '.join(sorted(annotators))}")

# Example usage
if __name__ == "__main__":
    MAIN_PATH = "/Users/liambarrett/Evident-AI/nlp_ehr/"
    INPUT_FILE = f"{MAIN_PATH}results/mongodb/mongodb_export_20250319_111312/reduced_users.json"
    OUTPUT_FILE = f"{MAIN_PATH}results/mongodb/mongodb_export_20250319_111312/fixed_users.json"

    # First analyze the current assignments
    analysis = analyze_letter_assignments(INPUT_FILE)

    # Then fix issues if needed
    if input("Would you like to fix the issues? (y/n): ").lower() == 'y':
        fix_letter_assignments(INPUT_FILE, OUTPUT_FILE, analysis)
        compare_assignments(INPUT_FILE, OUTPUT_FILE)

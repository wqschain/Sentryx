from app.services.sentiment.test_data import TEST_CASES
from app.services.sentiment.data_augmentation import get_balanced_dataset, augment_dataset
from collections import Counter

def print_separator():
    print("\n" + "="*50 + "\n")

# 1. Original Data Coverage
print("1. ORIGINAL DATA COVERAGE")
print_separator()
total_examples = 0
for category, cases in TEST_CASES.items():
    num_cases = len(cases)
    total_examples += num_cases
    print(f"{category:20}: {num_cases:3} examples")
    
    # Print sentiment distribution per category
    sentiments = Counter(case[1] for case in cases)
    print("   Sentiments:", dict(sentiments))
    print()

print(f"Total original examples: {total_examples}")

# 2. Augmented Data
print_separator()
print("2. AUGMENTED DATA STATISTICS")
print_separator()

augmented = augment_dataset()
total_augmented = 0
for category, cases in augmented.items():
    num_cases = len(cases)
    total_augmented += num_cases
    print(f"{category:20}: {num_cases:3} examples")
    
    # Print sentiment distribution per category
    sentiments = Counter(case[1] for case in cases)
    print("   Sentiments:", dict(sentiments))
    print()

print(f"Total augmented examples: {total_augmented}")
print(f"Augmentation multiplier: {total_augmented/total_examples:.1f}x")

# 3. Balanced Dataset
print_separator()
print("3. BALANCED DATASET STATISTICS")
print_separator()

balanced = get_balanced_dataset()
sentiments = Counter(case[1] for case in balanced)
print("Sentiment distribution in balanced dataset:")
print(dict(sentiments))
print(f"\nTotal balanced examples: {len(balanced)}") 
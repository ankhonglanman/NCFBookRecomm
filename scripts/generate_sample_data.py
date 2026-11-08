"""
Project: Book-NCF
Module: scripts.generate_sample_data
Description: Generates synthetic ratings.csv and books.csv for quick testing
             of the entire pipeline without needing external dataset downloads.
"""

import os
import random
import argparse
import pandas as pd


def generate_sample_data(num_users: int = 100, num_books: int = 200, interactions_per_user: int = 15, output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    # 1. Generate books metadata
    genres = ["Classic", "Sci-Fi", "Fantasy", "Mystery", "Non-Fiction", "Romance", "Thriller"]
    authors = ["Jane Austen", "George Orwell", "J.K. Rowling", "J.R.R. Tolkien", "Isaac Asimov", "Agatha Christie", "Stephen King"]
    
    books_data = []
    for book_id in range(1, num_books + 1):
        genre = random.choice(genres)
        author = random.choice(authors)
        title = f"{genre} Book #{book_id}"
        year = random.randint(1950, 2023)
        books_data.append({
            "book_id": book_id,
            "title": title,
            "authors": author,
            "publication_year": year,
            "genre": genre
        })

    books_df = pd.DataFrame(books_data)
    books_path = os.path.join(output_dir, "books.csv")
    books_df.to_csv(books_path, index=False)
    print(f"Generated {len(books_df)} books -> {books_path}")

    # 2. Generate ratings (power-law / popular books bias)
    popular_books = list(range(1, int(num_books * 0.2) + 1))
    all_books = list(range(1, num_books + 1))

    ratings_data = []
    for user_id in range(1, num_users + 1):
        # Sample interacted books (blend of popular + random)
        n_interact = random.randint(max(8, interactions_per_user - 5), interactions_per_user + 5)
        
        # 60% chance popular, 40% chance any book
        chosen_books = set()
        while len(chosen_books) < n_interact:
            if random.random() < 0.6 and popular_books:
                book = random.choice(popular_books)
            else:
                book = random.choice(all_books)
            chosen_books.add(book)

        for book_id in chosen_books:
            rating = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
            timestamp = 1600000000 + random.randint(0, 10000000)
            ratings_data.append({
                "user_id": user_id,
                "book_id": book_id,
                "rating": rating,
                "timestamp": timestamp
            })

    ratings_df = pd.DataFrame(ratings_data)
    ratings_path = os.path.join(output_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Generated {len(ratings_df)} ratings -> {ratings_path}")
    print("Sample data generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sample dataset for Book-NCF")
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--num_books", type=int, default=200)
    parser.add_argument("--interactions_per_user", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()

    generate_sample_data(args.num_users, args.num_books, args.interactions_per_user, args.output_dir)

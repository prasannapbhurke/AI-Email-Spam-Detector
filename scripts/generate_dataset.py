#!/usr/bin/env python3
"""
Generate synthetic spam/ham training dataset.
"""

import csv
import random

# Spam examples
SPAM_EXAMPLES = [
    ("free money click here now", "spam"),
    ("congratulations you won lottery", "spam"),
    ("urgent action required verify account", "spam"),
    ("cheap pharma online no prescription", "spam"),
    ("nigerian prince needs your help", "spam"),
    ("get rich quick work from home", "spam"),
    ("limited time offer act now", "spam"),
    ("your account has been compromised click link", "spam"),
    ("exclusive deal for you only today", "spam"),
    ("earn $5000 daily from home", "spam"),
    ("dear winner claim your prize", "spam"),
    ("suspended account verify now", "spam"),
    ("bitcoin investment opportunity guaranteed returns", "spam"),
    ("lose weight fast no exercise needed", "spam"),
    ("adult content exclusive access click", "spam"),
    ("credit card offer approved click", "spam"),
    ("meet singles in your area tonight", "spam"),
    ("job offer salary $100k work from home", "spam"),
    ("congratulations your email won prize", "spam"),
    ("order now free shipping", "spam"),
    ("extreme discount luxury items", "spam"),
    ("act immediately avoid account closure", "spam"),
    ("special promotion just for you", "spam"),
    ("million dollar opportunity awaits", "spam"),
    ("your refund is waiting claim now", "spam"),
]

# Ham examples
HAM_EXAMPLES = [
    ("team meeting tomorrow at 3pm", "ham"),
    ("project update the codebase is ready", "ham"),
    ("thanks for the update john", "ham"),
    ("lunch plans for friday", "ham"),
    ("can you review my pull request", "ham"),
    ("shipping confirmation order 12345", "ham"),
    (" receipt for your purchase", "ham"),
    ("password reset request received", "ham"),
    ("weekly report attached please review", "ham"),
    ("happy birthday hope you have a great day", "ham"),
    ("calendar invite team sync", "ham"),
    ("documents from meeting shared with you", "ham"),
    ("flight confirmation booking abc123", "ham"),
    ("subscription renewal coming up", "ham"),
    ("please review the proposal", "ham"),
    ("great work on the presentation", "ham"),
    ("dinner reservations for saturday", "ham"),
    ("new message in your inbox", "ham"),
    ("hotel booking confirmation ref 456", "ham"),
    ("reminder about appointment tomorrow", "ham"),
    ("thank you for your purchase", "ham"),
    ("your order has shipped", "ham"),
    ("welcome to our service", "ham"),
    ("account settings updated successfully", "ham"),
    ("monthly newsletter january", "ham"),
]

def generate_dataset(num_samples=2000, output_path="data/spam_dataset.csv"):
    """Generate a synthetic spam/ham dataset."""
    random.seed(42)

    # Base examples
    data = []

    # Add base examples multiple times with variations
    for _ in range(num_samples // 50):
        for text, label in SPAM_EXAMPLES:
            # Add random variations
            variations = [
                text,
                text.upper(),
                text.title(),
                f"{text}!!!",
                f"[SPAM] {text}",
                f"{text} http://example.com",
            ]
            data.append((random.choice(variations), label))

        for text, label in HAM_EXAMPLES:
            variations = [
                text,
                text.upper(),
                text.title(),
                f"Re: {text}",
                f"{text} - thanks",
            ]
            data.append((random.choice(variations), label))

    # Add more random samples
    spam_templates = [
        "free {} click {}",
        "{} urgent action {}",
        "congratulations {} won {}",
        "{} offer only {}",
        "your {} is {} click {}",
        "{} limited time {}",
        "earn {} daily {}",
        "act {} {} now",
        "exclusive {} for {}",
        "claim your {} {}",
    ]

    ham_templates = [
        "meeting {} at {}",
        "thanks for {}",
        "please {} the {}",
        "can you {}",
        "{} update attached",
        "your {} confirmation",
        "re: {}",
        "{} for {}",
        "happy {}",
        "reminder: {}",
    ]

    spam_words = ["money", "prize", "winner", "lottery", "offer", "deal", "cash", "bitcoin", "investment", "million", "free", "discount", "cheap", "order", "click", "link", "account", "verify", "suspended"]
    ham_words = ["lunch", "dinner", "meeting", "project", "team", "review", "document", "report", "thanks", "birthday", "appointment", "schedule", "confirm", "update", "sync", "welcome"]

    for _ in range(num_samples // 10):
        if random.random() < 0.5:
            template = random.choice(spam_templates)
            words = [random.choice(spam_words) for _ in range(3)]
            text = template.format(*words)
            data.append((text, "spam"))
        else:
            template = random.choice(ham_templates)
            words = [random.choice(ham_words) for _ in range(2)]
            text = template.format(*words)
            data.append((text, "ham"))

    random.shuffle(data)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in data[:num_samples]:
            writer.writerow([text, label])

    print(f"Generated {min(num_samples, len(data))} samples to {output_path}")

    # Print stats
    spam_count = sum(1 for _, l in data[:num_samples] if l == "spam")
    ham_count = num_samples - spam_count
    print(f"  Spam: {spam_count} ({spam_count/num_samples*100:.1f}%)")
    print(f"  Ham: {ham_count} ({ham_count/num_samples*100:.1f}%)")

if __name__ == "__main__":
    generate_dataset(num_samples=2000, output_path="data/spam_dataset.csv")

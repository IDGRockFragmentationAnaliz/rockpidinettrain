"""Валидация PQ: исключённая область маски становится белой границей."""

from scripts_pipeline.task4_validate import main as validate


def main() -> None:
    validate(use_pq=True)


if __name__ == "__main__":
    main()

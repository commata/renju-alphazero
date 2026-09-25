"""Two-player terminal game: python -m renju."""
from . import BLACK, Game, IllegalMove, SIZE


def display(game: Game) -> None:
    print("   " + " ".join(f"{c + 1:2}" for c in range(SIZE)))
    for row, cells in enumerate(game.board):
        print(f"{row + 1:2} " + "  ".join("●" if x == BLACK else "○" if x else "·" for x in cells))


def main() -> None:
    game = Game()
    print("렌주 대국 (흑 첫 수 중앙 고정, 이후 자유 착수). 행 열: '8 8', 되돌리기: u, 종료: q")
    while True:
        display(game)
        if game.done:
            print("승자: " + ("흑" if game.winner == BLACK else "백" if game.winner else "무승부"))
            return
        player = "흑" if game.to_play == BLACK else "백"
        try:
            command = input(f"{player} 착수 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n대국을 종료합니다")
            return
        if command == "q":
            return
        if command == "u":
            try:
                game.undo()
            except IllegalMove as exc:
                print(exc)
            continue
        try:
            row, col = map(int, command.split())
            game.play(row - 1, col - 1)
        except (ValueError, IllegalMove) as exc:
            print(f"착수할 수 없습니다: {exc}")


if __name__ == "__main__":
    main()

import chess


all_moves = []
squares = list(chess.SQUARES)

for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue

            # Add normal move
            all_moves.append(chess.Move(from_sq, to_sq))

            # Add promotion moves if desired
            if chess.square_rank(to_sq) in [0, 7]:
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    all_moves.append(chess.Move(from_sq, to_sq, promotion=promo))

            #Add drops (for variants like Crazyhouse)
            #if include_drops:
            #    for piece_type in range(1, 7):
            #        all_moves.append(chess.Move(None, to_sq, drop=piece_type))
print(len(all_moves))


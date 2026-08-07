"""
negation_augment.py
═══════════════════
Generates augmentation data targeting negation/polarity/double-negative
constructions that the model currently fails to translate correctly.

The model inverts meaning on patterns like:
    EN: "he is no less kind than his sister"
    → JA: "彼は妹より親切ではない" (WRONG — means "not kind")
    Should be: "彼は姉に劣らず親切だ" (equally kind)

This script creates template-based parallel pairs covering:
    - Double negatives (no less, no more, not unlike, etc.)
    - Comparative structures with negation
    - Complex negation patterns (hardly/scarcely...when)
    - Japanese double negation → English equivalents

Usage:
    python data/data-manage/negation_augment.py
    python data/data-manage/negation_augment.py --output data/dataset/train-set/negation_pairs.txt --oversample 8
"""

import os
import random
import argparse
import logging
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: "no less ... than" patterns (equally / at least as much)
# ──────────────────────────────────────────────────────────────────────────────

NO_LESS_THAN_PAIRS = [
    # (EN, JA primary, [JA variants])
    ("He is no less kind than his sister.", "彼は姉に劣らず親切だ。", ["彼は姉と同じくらい親切だ。"]),
    ("She is no less talented than her brother.", "彼女は兄に劣らず才能がある。", ["彼女は兄と同じくらい才能がある。"]),
    ("This book is no less interesting than that one.", "この本はあの本に劣らず面白い。", ["この本はあの本と同じくらい面白い。"]),
    ("He is no less brave than a lion.", "彼はライオンに劣らず勇敢だ。", ["彼はライオンと同じくらい勇敢だ。"]),
    ("She is no less beautiful than a flower.", "彼女は花に劣らず美しい。", []),
    ("This problem is no less important than the previous one.", "この問題は前の問題に劣らず重要だ。", []),
    ("He is no less intelligent than his father.", "彼は父親に劣らず賢い。", ["彼は父親と同じくらい頭がいい。"]),
    ("The movie is no less exciting than the book.", "映画は原作に劣らず面白い。", []),
    ("Her singing is no less wonderful than a professional's.", "彼女の歌はプロに劣らず素晴らしい。", []),
    ("This city is no less busy than Tokyo.", "この街は東京に劣らず忙しい。", ["この街は東京と同じくらい賑やかだ。"]),
    ("He works no less hard than anyone else.", "彼は誰にも劣らず一生懸命働く。", []),
    ("She is no less determined than her rival.", "彼女はライバルに劣らず決意が固い。", []),
    ("The result is no less surprising than expected.", "結果は予想に劣らず驚くべきものだ。", []),
    ("His effort is no less admirable than hers.", "彼の努力は彼女に劣らず立派だ。", []),
    ("This restaurant is no less popular than the one downtown.", "このレストランは街中のものに劣らず人気がある。", []),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 2: "no more ... than" patterns (equally NOT / neither ... nor)
# ──────────────────────────────────────────────────────────────────────────────

NO_MORE_THAN_PAIRS = [
    ("He is no more a genius than I am.", "彼が天才でないのは私と同じだ。", ["彼も私も天才ではない。"]),
    ("She can no more swim than a cat can.", "彼女は猫と同様に泳げない。", ["彼女は泳げるはずがない。"]),
    ("He is no more fit for the job than a child.", "彼がその仕事に向いていないのは子供と同じだ。", []),
    ("This is no more true than saying the earth is flat.", "これは地球が平らだと言うのと同じくらい嘘だ。", []),
    ("She is no more happy than her sister.", "彼女も姉も幸せではない。", ["彼女が幸せでないのは姉と同じだ。"]),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 3: "not un-" patterns (litotes — double negation = mild positive)
# ──────────────────────────────────────────────────────────────────────────────

NOT_UN_PAIRS = [
    ("It is not unlikely that he will come.", "彼が来る可能性は十分ある。", ["彼が来ることはありえなくはない。"]),
    ("She is not unkind to strangers.", "彼女は知らない人に対しても親切だ。", ["彼女は見知らぬ人にも冷たくない。"]),
    ("The result was not unexpected.", "その結果は予想通りだった。", ["その結果は意外ではなかった。"]),
    ("His success is not undeserved.", "彼の成功は当然のことだ。", ["彼は成功に値する。"]),
    ("It's not unreasonable to expect that.", "それを期待するのは無理もない。", ["それを期待するのは妥当だ。"]),
    ("The proposal is not without merit.", "その提案にはメリットがある。", ["その提案には利点がないわけではない。"]),
    ("He was not unaware of the problem.", "彼はその問題に気づいていた。", ["彼はその問題を知っていた。"]),
    ("The task is not impossible.", "その課題は不可能ではない。", ["やれないことはない。"]),
    ("She was not unhappy with the result.", "彼女はその結果に満足していた。", ["彼女は結果を悪く思っていなかった。"]),
    ("It is not uncommon to see that.", "それを見ることは珍しくない。", ["それはよくあることだ。"]),
    ("His behavior is not unusual.", "彼の行動は普通だ。", ["彼の行動は珍しくない。"]),
    ("The price is not unreasonable.", "その値段は妥当だ。", ["その価格は高すぎるわけではない。"]),
    ("Her reaction was not unjustified.", "彼女の反応は正当だった。", ["彼女の反応は無理もなかった。"]),
    ("It was not unforeseen.", "それは予測できたことだった。", ["想定外ではなかった。"]),
    ("The delay is not insignificant.", "その遅延は無視できない。", ["その遅延はかなりのものだ。"]),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 4: "can't help but" / "can't help ~ing" patterns
# ──────────────────────────────────────────────────────────────────────────────

CANT_HELP_PAIRS = [
    ("I can't help but laugh.", "笑わずにはいられない。", ["つい笑ってしまう。"]),
    ("She can't help worrying about him.", "彼女は彼のことが心配でたまらない。", ["彼女は彼のことを心配せずにいられない。"]),
    ("I couldn't help but notice.", "気づかずにはいられなかった。", ["どうしても目についた。"]),
    ("He can't help feeling nervous.", "彼は緊張せずにはいられない。", ["彼はどうしても緊張してしまう。"]),
    ("I can't help thinking about it.", "そのことを考えずにはいられない。", ["どうしてもそのことが頭から離れない。"]),
    ("She couldn't help crying.", "彼女は泣かずにいられなかった。", ["彼女はどうしても泣いてしまった。"]),
    ("I can't help but admire her courage.", "彼女の勇気に感心せずにはいられない。", []),
    ("He can't help being late.", "彼はどうしても遅刻してしまう。", []),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 5: "hardly / scarcely ... when" / "no sooner ... than"
# ──────────────────────────────────────────────────────────────────────────────

HARDLY_WHEN_PAIRS = [
    ("Hardly had he arrived when it started raining.", "彼が到着するやいなや雨が降り始めた。", ["彼が着いた途端に雨が降り出した。"]),
    ("No sooner had she left than the phone rang.", "彼女が出かけた途端に電話が鳴った。", ["彼女が去るやいなや電話が鳴った。"]),
    ("Scarcely had I sat down when someone knocked.", "座った途端に誰かがノックした。", ["座るか座らないかのうちにノックが聞こえた。"]),
    ("Hardly had the game begun when it started to rain.", "試合が始まるやいなや雨が降り出した。", []),
    ("No sooner had I opened the door than the cat ran out.", "ドアを開けた途端に猫が飛び出した。", []),
    ("Hardly had we finished eating when the bill arrived.", "食べ終わるやいなや請求書が来た。", []),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 6: "neither ... nor" / "not ... either"
# ──────────────────────────────────────────────────────────────────────────────

NEITHER_NOR_PAIRS = [
    ("Neither he nor she agreed.", "彼も彼女も同意しなかった。", []),
    ("I like neither coffee nor tea.", "コーヒーも紅茶もどちらも好きではない。", []),
    ("She speaks neither English nor French.", "彼女は英語もフランス語も話せない。", []),
    ("Neither the teacher nor the students were happy.", "先生も生徒も満足していなかった。", []),
    ("He can neither read nor write.", "彼は読むことも書くこともできない。", []),
    ("Neither rain nor snow stopped them.", "雨も雪も彼らを止められなかった。", []),
    ("I don't like it either.", "私もそれは好きではない。", []),
    ("She didn't come either.", "彼女も来なかった。", []),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 7: Complex comparative negation patterns
# ──────────────────────────────────────────────────────────────────────────────

COMPLEX_COMPARATIVE_PAIRS = [
    ("Nothing is more important than health.", "健康ほど大切なものはない。", ["健康が最も大切だ。"]),
    ("Nothing is as valuable as time.", "時間ほど貴重なものはない。", ["時間が最も価値がある。"]),
    ("I have never seen a more beautiful sunset.", "あんなに美しい夕日は見たことがない。", ["あれほど美しい夕日を見たことはなかった。"]),
    ("There is nothing he can't do.", "彼にできないことはない。", ["彼は何でもできる。"]),
    ("She is the last person to tell a lie.", "彼女は嘘をつくような人では決してない。", ["彼女は最も嘘をつかない人だ。"]),
    ("It is anything but easy.", "それは決して簡単ではない。", ["それは簡単どころではない。"]),
    ("He is far from perfect.", "彼は完璧とはほど遠い。", ["彼は完璧には程遠い。"]),
    ("The exam was anything but fair.", "その試験は公平とは言えなかった。", ["その試験は公平とはほど遠かった。"]),
    ("I could not agree more.", "まったく同感だ。", ["これ以上ないほど賛成だ。"]),
    ("She was none the wiser for the explanation.", "説明を聞いても彼女はさっぱりわからなかった。", []),
    ("He is nothing if not careful.", "彼は何よりも慎重だ。", ["彼はとにかく慎重な人だ。"]),
    ("The food was not bad at all.", "その料理は悪くなかった。", ["むしろ美味しかった。"]),
    ("I would not say no to that.", "それなら喜んで受けよう。", ["断る理由はない。"]),
    ("That's not to say it was perfect.", "完璧だったとは言わないが。", ["完璧だったわけではないが。"]),
    ("He is second to none in this field.", "彼はこの分野では誰にも引けを取らない。", ["彼はこの分野で最も優れている。"]),
]

# ──────────────────────────────────────────────────────────────────────────────
# Section 8: Japanese negation patterns → English
# ──────────────────────────────────────────────────────────────────────────────

JA_NEGATION_PAIRS = [
    ("He is not unkind.", "彼が不親切なわけではない。", ["彼は親切だ。"]),
    ("It's not that I don't like it.", "嫌いなわけではない。", ["嫌いというわけではない。"]),
    ("I can't say I disagree.", "反対だとは言えない。", ["ある程度賛成だ。"]),
    ("It is not without reason.", "理由がないわけではない。", ["それには理由がある。"]),
    ("Not a day goes by without thinking of you.", "あなたのことを考えない日はない。", ["毎日あなたのことを思っている。"]),
    ("You shouldn't underestimate him.", "彼を侮ってはいけない。", ["彼を甘く見るな。"]),
    ("It won't do you any harm.", "損にはならない。", ["やって損はない。"]),
    ("I never said I didn't want to go.", "行きたくないとは言っていない。", ["行きたくないなんて言った覚えはない。"]),
    ("There is no telling what will happen.", "何が起こるか分からない。", ["予測不可能だ。"]),
    ("He left no stone unturned.", "彼はあらゆる手を尽くした。", ["彼は全力を尽くした。"]),
    ("It is by no means certain.", "それは決して確実ではない。", ["それはまったく確実ではない。"]),
    ("She is not without talent.", "彼女に才能がないわけではない。", ["彼女には才能がある。"]),
    ("I would be lying if I said I wasn't worried.", "心配していないと言ったら嘘になる。", []),
    ("There is nothing I wouldn't do for you.", "あなたのためならなんでもする。", []),
    ("You can't be too careful.", "注意しすぎることはない。", ["いくら注意してもしすぎることはない。"]),
]


# ──────────────────────────────────────────────────────────────────────────────
# Combine all sections
# ──────────────────────────────────────────────────────────────────────────────

ALL_NEGATION_SECTIONS = [
    ("no_less_than", NO_LESS_THAN_PAIRS),
    ("no_more_than", NO_MORE_THAN_PAIRS),
    ("not_un", NOT_UN_PAIRS),
    ("cant_help", CANT_HELP_PAIRS),
    ("hardly_when", HARDLY_WHEN_PAIRS),
    ("neither_nor", NEITHER_NOR_PAIRS),
    ("complex_comparative", COMPLEX_COMPARATIVE_PAIRS),
    ("ja_negation", JA_NEGATION_PAIRS),
]


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_negation_corpus(
    output_path: str,
    oversample: int = 8,
    seed: int = 42,
    synthetic_score: float = 0.95,
):
    """
    Generate negation/polarity augmentation data.

    Parameters
    ----------
    output_path : str
        Output TSV file path.
    oversample : int
        Repetition factor for each generated pair.
    seed : int
        Random seed.
    synthetic_score : float
        Synthetic BiCleaner score.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    all_pairs = []
    section_counts = {}

    for section_name, pairs in ALL_NEGATION_SECTIONS:
        count = 0
        for en, ja_primary, ja_variants in pairs:
            all_pairs.append((en, ja_primary))
            count += 1
            for ja_var in ja_variants:
                all_pairs.append((en, ja_var))
                count += 1
        section_counts[section_name] = count

    # Deduplicate
    all_pairs = list(set(all_pairs))

    # Oversample
    expanded = all_pairs * oversample
    random.shuffle(expanded)

    # Write in training data format
    total_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for en, ja in expanded:
            line = f"negation_aug\tnegation_aug\t{synthetic_score:.3f}\t{en}\t{ja}\n"
            f.write(line)
            total_written += 1

    logger.info("Generated %d negation augmentation pairs → %s", total_written, output_path)
    logger.info("  Unique pairs: %d", len(all_pairs))
    logger.info("  Oversample factor: %d×", oversample)
    for section, count in section_counts.items():
        logger.info("    %s: %d pairs", section, count)

    return total_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate negation/polarity augmentation data for EN↔JA.")
    p.add_argument("--output", default="data/dataset/train-set/negation_pairs.txt",
                   help="Output TSV file path")
    p.add_argument("--oversample", type=int, default=8,
                   help="Repetition factor (default: 8)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--score", type=float, default=0.95,
                   help="Synthetic BiCleaner score (default: 0.95)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_negation_corpus(
        output_path=args.output,
        oversample=args.oversample,
        seed=args.seed,
        synthetic_score=args.score,
    )

"""
idiom_augmentation.py
═════════════════════
Generates a curated parallel corpus of English↔Japanese idioms and
figurative expressions to teach the translation model non-literal mappings.

The model currently translates idioms literally (e.g., "break a leg" → "脚を折れ")
because the web-crawled training data lacks sufficient idiomatic examples.
This script outputs pairs in the same TSV format as the training data.

Usage:
    python data/data-manage/idiom_augmentation.py
    python data/data-manage/idiom_augmentation.py --output data/dataset/train-set/idiom_pairs.txt --oversample 10
"""

import os
import random
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Idiom Dictionary
# ──────────────────────────────────────────────────────────────────────────────
# Format: (English idiom, Japanese natural translation, [optional variations])
# Each entry may include multiple Japanese translations for different registers.
# The English side may also have variations to improve generalization.

IDIOM_PAIRS = [
    # ── Common English Idioms → Natural Japanese ──
    ("break a leg", "成功を祈る", ["うまくいくことを祈ってるよ", "がんばって"]),
    ("it's raining cats and dogs", "土砂降りだ", ["大雨が降っている", "ひどい雨だ"]),
    ("kill two birds with one stone", "一石二鳥だ", ["一石二鳥を狙う"]),
    ("the ball is in your court", "あとはあなた次第だ", ["判断はあなたに任せる"]),
    ("bite the bullet", "覚悟を決める", ["腹をくくる", "思い切ってやる"]),
    ("hit the nail on the head", "的を射ている", ["まさにその通りだ", "核心をついている"]),
    ("let the cat out of the bag", "秘密を漏らしてしまう", ["うっかり秘密を話す"]),
    ("once in a blue moon", "めったにない", ["ごくまれに", "滅多にないことだ"]),
    ("piece of cake", "朝飯前だ", ["簡単なことだ", "楽勝だ"]),
    ("speak of the devil", "噂をすれば影がさす", ["噂をすれば"]),
    ("the last straw", "堪忍袋の緒が切れた", ["我慢の限界だ", "もう限界だ"]),
    ("barking up the wrong tree", "見当違いだ", ["お門違いだ", "間違った方向を向いている"]),
    ("beat around the bush", "遠回しに言う", ["回りくどく言う"]),
    ("burn the midnight oil", "夜遅くまで勉強する", ["夜なべをする", "徹夜で作業する"]),
    ("cost an arm and a leg", "非常に高い", ["法外な値段だ", "目が飛び出るほど高い"]),
    ("cry over spilled milk", "覆水盆に返らず", ["過ぎたことを嘆いても仕方ない"]),
    ("get cold feet", "怖じ気づく", ["二の足を踏む", "尻込みする"]),
    ("go the extra mile", "一層の努力をする", ["期待以上のことをする"]),
    ("have a change of heart", "気が変わる", ["考えを改める", "心変わりする"]),
    ("in hot water", "窮地に立たされている", ["まずい状況だ", "トラブルに巻き込まれた"]),
    ("it takes two to tango", "喧嘩両成敗だ", ["お互い様だ", "双方に責任がある"]),
    ("jump on the bandwagon", "流行に乗る", ["便乗する", "時流に乗る"]),
    ("keep your chin up", "元気を出して", ["くじけないで", "がんばれ"]),
    ("miss the boat", "チャンスを逃す", ["機会を逃した", "乗り遅れた"]),
    ("not my cup of tea", "私の好みではない", ["私には合わない"]),
    ("on cloud nine", "有頂天だ", ["天にも昇る気持ちだ", "最高に幸せだ"]),
    ("pull someone's leg", "からかう", ["冗談を言う", "ふざける"]),
    ("see eye to eye", "意見が一致する", ["同じ考えだ"]),
    ("sit on the fence", "どっちつかずだ", ["態度を決めかねている", "日和見している"]),
    ("steal someone's thunder", "人の手柄を横取りする", ["お株を奪う"]),
    ("take it with a grain of salt", "話半分に聞く", ["鵜呑みにしない"]),
    ("the elephant in the room", "見て見ぬふりをしている問題", ["誰もが気づいている問題"]),
    ("throw in the towel", "降参する", ["あきらめる", "ギブアップする"]),
    ("under the weather", "体調が悪い", ["具合が悪い", "風邪気味だ"]),
    ("when pigs fly", "ありえない", ["絶対に起こらない"]),
    ("whole nine yards", "全部", ["何もかも", "すべて"]),
    ("you can't judge a book by its cover", "人は見かけによらない", ["外見で判断してはいけない"]),
    ("actions speak louder than words", "行動は言葉より雄弁だ", ["口先だけではだめだ"]),
    ("add fuel to the fire", "火に油を注ぐ", ["事態を悪化させる"]),
    ("back to square one", "振り出しに戻る", ["最初からやり直しだ"]),
    ("better late than never", "遅くてもやらないよりましだ", ["遅れてもやらないよりはいい"]),
    ("blow off steam", "ストレスを発散する", ["うっぷんを晴らす", "気晴らしをする"]),
    ("call it a day", "今日はここまでにする", ["切り上げよう", "今日は終わりにしよう"]),
    ("cross that bridge when you come to it", "その時はその時だ", ["その時になったら考えよう"]),
    ("curiosity killed the cat", "好奇心は身を滅ぼす", ["詮索はほどほどに"]),
    ("cut to the chase", "本題に入る", ["要点を言う", "結論から言う"]),
    ("don't put all your eggs in one basket", "一つのことに全てを賭けるな", ["リスクを分散させろ"]),
    ("easier said than done", "言うは易く行うは難し", ["口で言うのは簡単だ"]),
    ("every cloud has a silver lining", "どんな不幸にも良い面がある", ["禍を転じて福となす"]),
    ("get out of hand", "手に負えなくなる", ["収拾がつかなくなる"]),
    ("give the benefit of the doubt", "善意に解釈する", ["疑わしきは罰せず"]),
    ("go back to the drawing board", "最初からやり直す", ["一から出直す"]),
    ("hang in there", "頑張れ", ["あきらめないで", "持ちこたえて"]),
    ("hit the books", "猛勉強する", ["一生懸命勉強する"]),
    ("in a nutshell", "要するに", ["一言で言えば", "簡単に言うと"]),
    ("it's not rocket science", "そんなに難しいことじゃない", ["誰でもできることだ"]),
    ("keep an eye on", "目を離さない", ["注意して見ている", "監視する"]),
    ("leave no stone unturned", "あらゆる手を尽くす", ["全力を尽くす"]),
    ("let bygones be bygones", "過去のことは水に流そう", ["過ぎたことは忘れよう"]),
    ("make a long story short", "手短に言えば", ["要約すると"]),
    ("no pain no gain", "苦労なくして成果なし", ["努力なしに成功はない"]),
    ("on the same page", "同じ認識を持っている", ["意思疎通ができている"]),
    ("play it by ear", "臨機応変にやる", ["その場の判断で決める"]),
    ("practice makes perfect", "習うより慣れろ", ["練習すれば完璧になる"]),
    ("pull yourself together", "しっかりしろ", ["気を取り直して", "落ち着いて"]),
    ("read between the lines", "行間を読む", ["言外の意味を読み取る"]),
    ("sleep on it", "一晩考えてみる", ["よく考えてから決める"]),
    ("spill the beans", "秘密をばらす", ["うっかり口を滑らせる"]),
    ("take a rain check", "また今度にする", ["次の機会にする"]),
    ("the apple doesn't fall far from the tree", "蛙の子は蛙", ["親に似る"]),
    ("think outside the box", "既成概念にとらわれない", ["型にはまらない発想をする"]),
    ("time flies", "光陰矢の如し", ["時間が経つのは早い", "あっという間だ"]),
    ("two heads are better than one", "三人寄れば文殊の知恵", ["二人で考えた方がいい"]),
    ("wrap your head around", "理解しようとする", ["頭を整理する"]),

    # ── Common Japanese Idioms → Natural English ──
    # (These are stored reversed for JA→EN training)
    ("a drop in the bucket", "焼け石に水", []),
    ("tip of the iceberg", "氷山の一角", []),
    ("out of the frying pan into the fire", "一難去ってまた一難", []),
    ("birds of a feather flock together", "類は友を呼ぶ", []),
    ("even monkeys fall from trees", "猿も木から落ちる", []),
    ("seeing is believing", "百聞は一見にしかず", []),
    ("the early bird catches the worm", "早起きは三文の徳", []),
    ("strike while the iron is hot", "鉄は熱いうちに打て", []),
    ("when in Rome do as the Romans do", "郷に入っては郷に従え", []),
    ("still waters run deep", "能ある鷹は爪を隠す", []),
    ("haste makes waste", "急がば回れ", []),
    ("many hands make light work", "人手が多ければ仕事は楽になる", []),
    ("out of sight out of mind", "去る者は日日に疎し", []),
    ("a picture is worth a thousand words", "百聞は一見に如かず", []),
    ("the pen is mightier than the sword", "ペンは剣よりも強し", []),
    ("don't count your chickens before they hatch", "捕らぬ狸の皮算用", []),
    ("there's no use crying over spilled milk", "覆水盆に返らず", []),
    ("rome wasn't built in a day", "ローマは一日にして成らず", []),
    ("necessity is the mother of invention", "必要は発明の母", []),
    ("where there's a will there's a way", "意志あるところに道は開ける", []),
]


# ──────────────────────────────────────────────────────────────────────────────
# Contextual Sentence Templates
# ──────────────────────────────────────────────────────────────────────────────
# Wrap idioms in natural sentence context to help the model learn them
# in realistic settings rather than as isolated phrases.

EN_TEMPLATES = [
    "{idiom}.",
    "I think we should {idiom}.",
    "She told me to {idiom}.",
    "He always says \"{idiom}\".",
    "As the saying goes, {idiom}.",
    "You know what they say: {idiom}.",
    "In this situation, I would say {idiom}.",
    "My grandmother used to say, \"{idiom}\".",
    "Well, {idiom}, as they say.",
    "It's like the old saying: {idiom}.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_idiom_corpus(
    output_path: str,
    oversample: int = 5,
    seed: int = 42,
    synthetic_score: float = 0.95,
):
    """
    Generate a parallel corpus of idiomatic expression pairs.

    Parameters
    ----------
    output_path : str
        Output TSV file path (same format as training data).
    oversample : int
        Number of times each base pair is repeated (with shuffling).
    seed : int
        Random seed for reproducibility.
    synthetic_score : float
        Synthetic BiCleaner score assigned to generated pairs.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    all_pairs = []

    for en_idiom, ja_primary, ja_variants in IDIOM_PAIRS:
        # Primary pair (bare idiom)
        all_pairs.append((en_idiom, ja_primary))

        # Variant pairs
        for ja_var in ja_variants:
            all_pairs.append((en_idiom, ja_var))

    # Oversample by repeating and shuffling
    expanded = all_pairs * oversample
    random.shuffle(expanded)

    # Write in training data format
    # Format: dummy_source\tdummy_source\tscore\ten_text\tja_text
    total_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for en, ja in expanded:
            line = f"idiom_aug\tidiom_aug\t{synthetic_score:.3f}\t{en}\t{ja}\n"
            f.write(line)
            total_written += 1

    logger.info("Generated %d idiom pairs → %s", total_written, output_path)
    logger.info("  Base pairs: %d (with variants)", len(all_pairs))
    logger.info("  Oversample factor: %d×", oversample)

    return total_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate EN-JA idiom augmentation corpus.")
    p.add_argument("--output", default="data/dataset/train-set/idiom_pairs.txt",
                   help="Output TSV file path")
    p.add_argument("--oversample", type=int, default=5,
                   help="Number of times to repeat each pair (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--score", type=float, default=0.95,
                   help="Synthetic BiCleaner score (default: 0.95)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_idiom_corpus(
        output_path=args.output,
        oversample=args.oversample,
        seed=args.seed,
        synthetic_score=args.score,
    )

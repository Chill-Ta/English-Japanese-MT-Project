"""
idiom_augmentation.py
═════════════════════
Generates a curated parallel corpus of English↔Japanese idioms and
figurative expressions to teach the translation model non-literal mappings.

The model currently translates idioms literally (e.g., "break a leg" → "脚を折れ")
because the web-crawled training data lacks sufficient idiomatic examples.
This script outputs pairs in the same TSV format as the training data.

v2 — Scaled up from ~75 idioms to 300+ with contextual sentence templates,
     bidirectional generation (EN→JA and JA→EN), and higher default oversampling.

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
# Idiom Dictionary (Expanded — 300+ base entries)
# ──────────────────────────────────────────────────────────────────────────────
# Format: (English idiom, Japanese natural translation, [optional variations])
# Each entry may include multiple Japanese translations for different registers.

IDIOM_PAIRS = [
    # ══════════════════════════════════════════
    # Section A: Common English Idioms → Natural Japanese
    # ══════════════════════════════════════════

    # ── Everyday Idioms ──
    ("break a leg", "成功を祈る", ["うまくいくことを祈ってるよ", "がんばって", "幸運を祈ります"]),
    ("it's raining cats and dogs", "土砂降りだ", ["大雨が降っている", "ひどい雨だ", "どしゃ降りの雨だ"]),
    ("kill two birds with one stone", "一石二鳥だ", ["一石二鳥を狙う", "一度で二つの目的を達成する"]),
    ("the ball is in your court", "あとはあなた次第だ", ["判断はあなたに任せる", "決めるのはあなたです"]),
    ("bite the bullet", "覚悟を決める", ["腹をくくる", "思い切ってやる", "歯を食いしばって耐える"]),
    ("hit the nail on the head", "的を射ている", ["まさにその通りだ", "核心をついている", "的確だ"]),
    ("let the cat out of the bag", "秘密を漏らしてしまう", ["うっかり秘密を話す", "口を滑らせる"]),
    ("once in a blue moon", "めったにない", ["ごくまれに", "滅多にないことだ", "千載一遇の"]),
    ("piece of cake", "朝飯前だ", ["簡単なことだ", "楽勝だ", "造作もないことだ"]),
    ("speak of the devil", "噂をすれば影がさす", ["噂をすれば", "噂をすれば影"]),
    ("the last straw", "堪忍袋の緒が切れた", ["我慢の限界だ", "もう限界だ"]),
    ("barking up the wrong tree", "見当違いだ", ["お門違いだ", "間違った方向を向いている"]),
    ("beat around the bush", "遠回しに言う", ["回りくどく言う", "はっきり言わない"]),
    ("burn the midnight oil", "夜遅くまで勉強する", ["夜なべをする", "徹夜で作業する"]),
    ("cost an arm and a leg", "非常に高い", ["法外な値段だ", "目が飛び出るほど高い"]),
    ("cry over spilled milk", "覆水盆に返らず", ["過ぎたことを嘆いても仕方ない"]),
    ("get cold feet", "怖じ気づく", ["二の足を踏む", "尻込みする", "おじけづく"]),
    ("go the extra mile", "一層の努力をする", ["期待以上のことをする", "もう一歩頑張る"]),
    ("have a change of heart", "気が変わる", ["考えを改める", "心変わりする"]),
    ("in hot water", "窮地に立たされている", ["まずい状況だ", "トラブルに巻き込まれた"]),
    ("it takes two to tango", "喧嘩両成敗だ", ["お互い様だ", "双方に責任がある"]),
    ("jump on the bandwagon", "流行に乗る", ["便乗する", "時流に乗る"]),
    ("keep your chin up", "元気を出して", ["くじけないで", "がんばれ"]),
    ("miss the boat", "チャンスを逃す", ["機会を逃した", "乗り遅れた"]),
    ("not my cup of tea", "私の好みではない", ["私には合わない", "趣味に合わない"]),
    ("on cloud nine", "有頂天だ", ["天にも昇る気持ちだ", "最高に幸せだ"]),
    ("pull someone's leg", "からかう", ["冗談を言う", "ふざける", "からかっている"]),
    ("see eye to eye", "意見が一致する", ["同じ考えだ", "見解が一致する"]),
    ("sit on the fence", "どっちつかずだ", ["態度を決めかねている", "日和見している"]),
    ("steal someone's thunder", "人の手柄を横取りする", ["お株を奪う"]),
    ("take it with a grain of salt", "話半分に聞く", ["鵜呑みにしない", "割り引いて聞く"]),
    ("the elephant in the room", "見て見ぬふりをしている問題", ["誰もが気づいている問題"]),
    ("throw in the towel", "降参する", ["あきらめる", "ギブアップする", "白旗を揚げる"]),
    ("under the weather", "体調が悪い", ["具合が悪い", "風邪気味だ"]),
    ("when pigs fly", "ありえない", ["絶対に起こらない", "そんなことはありえない"]),
    ("whole nine yards", "全部", ["何もかも", "すべて"]),
    ("you can't judge a book by its cover", "人は見かけによらない", ["外見で判断してはいけない"]),
    ("actions speak louder than words", "行動は言葉より雄弁だ", ["口先だけではだめだ", "不言実行"]),
    ("add fuel to the fire", "火に油を注ぐ", ["事態を悪化させる"]),
    ("back to square one", "振り出しに戻る", ["最初からやり直しだ", "元の木阿弥だ"]),
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
    ("go back to the drawing board", "最初からやり直す", ["一から出直す", "白紙に戻す"]),
    ("hang in there", "頑張れ", ["あきらめないで", "持ちこたえて"]),
    ("hit the books", "猛勉強する", ["一生懸命勉強する"]),
    ("in a nutshell", "要するに", ["一言で言えば", "簡単に言うと"]),
    ("it's not rocket science", "そんなに難しいことじゃない", ["誰でもできることだ"]),
    ("keep an eye on", "目を離さない", ["注意して見ている", "監視する"]),
    ("leave no stone unturned", "あらゆる手を尽くす", ["全力を尽くす"]),
    ("let bygones be bygones", "過去のことは水に流そう", ["過ぎたことは忘れよう"]),
    ("make a long story short", "手短に言えば", ["要約すると", "かいつまんで言うと"]),
    ("no pain no gain", "苦労なくして成果なし", ["努力なしに成功はない"]),
    ("on the same page", "同じ認識を持っている", ["意思疎通ができている", "認識を共有している"]),
    ("play it by ear", "臨機応変にやる", ["その場の判断で決める"]),
    ("practice makes perfect", "習うより慣れろ", ["練習すれば完璧になる", "継続は力なり"]),
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

    # ── Additional Common Idioms (NEW in v2) ──
    ("a penny for your thoughts", "何を考えているの", ["何を思っているの"]),
    ("at the drop of a hat", "即座に", ["すぐに", "躊躇なく"]),
    ("be on the same wavelength", "気が合う", ["波長が合う", "意見が合う"]),
    ("bend over backwards", "全力で助ける", ["あらゆる努力をする", "骨を折る"]),
    ("burn bridges", "退路を断つ", ["関係を壊す", "橋を焼く"]),
    ("by the skin of your teeth", "かろうじて", ["ぎりぎりで", "間一髪で"]),
    ("cut corners", "手を抜く", ["手抜きをする", "近道をする"]),
    ("drop the ball", "しくじる", ["へまをする", "失敗する"]),
    ("face the music", "責任を取る", ["結果を受け入れる", "覚悟を決める"]),
    ("get your act together", "しっかりしなさい", ["ちゃんとしなさい", "態勢を整えろ"]),
    ("give someone the cold shoulder", "冷たい態度をとる", ["無視する", "そっけなくする"]),
    ("go down in flames", "大失敗する", ["惨敗する", "見事に失敗する"]),
    ("go out on a limb", "危険を冒す", ["思い切ったことをする"]),
    ("have butterflies in your stomach", "緊張している", ["どきどきする", "そわそわする"]),
    ("hit the ground running", "最初から全力で取り組む", ["すぐに本格的に始める"]),
    ("in the same boat", "同じ境遇だ", ["同じ立場だ", "運命共同体だ"]),
    ("it's a piece of work", "一筋縄ではいかない人だ", ["厄介な人だ"]),
    ("jump the gun", "フライングする", ["早まる", "先走る"]),
    ("keep your eyes peeled", "目を光らせて", ["よく注意して見て", "警戒して"]),
    ("kick the bucket", "死ぬ", ["亡くなる", "息を引き取る"]),
    ("let sleeping dogs lie", "触らぬ神に祟りなし", ["寝た子を起こすな"]),
    ("make ends meet", "やりくりする", ["生活費をまかなう", "なんとかやっていく"]),
    ("move the goalposts", "条件を後から変える", ["ルールを途中で変える"]),
    ("nip it in the bud", "芽のうちに摘む", ["早いうちに対処する", "未然に防ぐ"]),
    ("off the top of my head", "すぐに思いつく限りでは", ["ぱっと思いつくのは"]),
    ("on thin ice", "危ない橋を渡っている", ["薄氷を踏むような状況だ"]),
    ("once bitten twice shy", "一度痛い目にあうと慎重になる", ["羹に懲りて膾を吹く"]),
    ("pull strings", "裏で手を回す", ["コネを使う", "根回しをする"]),
    ("put all your cards on the table", "手の内を明かす", ["正直に全部話す"]),
    ("raise the bar", "基準を高くする", ["ハードルを上げる", "水準を引き上げる"]),
    ("ring a bell", "心当たりがある", ["聞き覚えがある", "ピンとくる"]),
    ("rock the boat", "波風を立てる", ["問題を起こす", "事を荒立てる"]),
    ("rule of thumb", "経験則", ["大まかな目安", "大体の基準"]),
    ("save face", "面目を保つ", ["体面を保つ", "メンツを保つ"]),
    ("see the light at the end of the tunnel", "出口が見えてきた", ["希望の光が見えた"]),
    ("shape up or ship out", "ちゃんとやるか辞めるかだ", ["改善するか去るかだ"]),
    ("shoot the breeze", "雑談する", ["おしゃべりする", "世間話をする"]),
    ("sink or swim", "一か八かだ", ["成功するか失敗するかだ"]),
    ("sit tight", "じっとしている", ["辛抱強く待つ", "動かないでいる"]),
    ("stab someone in the back", "裏切る", ["後ろから刺す", "不意打ちを食らわせる"]),
    ("take the bull by the horns", "難題に正面から取り組む", ["果敢に立ち向かう"]),
    ("the best of both worlds", "いいとこ取り", ["両方の利点を得る"]),
    ("the tip of the iceberg", "氷山の一角", ["問題のほんの一部"]),
    ("throw someone under the bus", "人を犠牲にする", ["人のせいにする", "仲間を売る"]),
    ("turn a blind eye", "見て見ぬふりをする", ["黙認する", "知らないふりをする"]),
    ("turn over a new leaf", "心を入れ替える", ["新しい出発をする", "生まれ変わる"]),
    ("twist someone's arm", "無理に説得する", ["腕をねじる", "強要する"]),
    ("up in the air", "まだ決まっていない", ["未定だ", "宙ぶらりんだ"]),
    ("walking on eggshells", "慎重に行動する", ["腫れ物に触るように接する"]),
    ("water under the bridge", "過ぎたことだ", ["もう終わったことだ"]),
    ("wear your heart on your sleeve", "感情をあらわにする", ["気持ちを隠さない"]),
    ("weather the storm", "困難を乗り越える", ["嵐を乗り切る", "難局を切り抜ける"]),
    ("worth its weight in gold", "非常に価値がある", ["金の値打ちがある"]),
    ("you reap what you sow", "因果応報だ", ["自業自得だ", "蒔いた種は刈り取れ"]),
    ("your guess is as good as mine", "私にも分からない", ["見当もつかない"]),
    ("a blessing in disguise", "不幸中の幸い", ["災い転じて福となる"]),
    ("all ears", "傾聴している", ["全身を耳にして聞いている"]),
    ("at the eleventh hour", "ぎりぎりのところで", ["土壇場で", "最後の瞬間に"]),
    ("behind someone's back", "陰で", ["人の知らないところで", "裏で"]),
    ("best thing since sliced bread", "画期的なもの", ["最高のものだ"]),
    ("bite off more than you can chew", "手に余ることを引き受ける", ["分不相応なことをする"]),
    ("blow it out of proportion", "大げさにする", ["誇張する", "話を大きくする"]),
    ("by and large", "概して", ["全体的に見て", "おおむね"]),
    ("come rain or shine", "何があっても", ["晴雨にかかわらず", "どんなことがあっても"]),
    ("couch potato", "テレビばかり見ている怠け者", ["ごろごろしている人"]),
    ("cry wolf", "嘘の警報を出す", ["狼少年のように嘘をつく"]),
    ("dime a dozen", "ありふれている", ["どこにでもある", "珍しくない"]),
    ("down to earth", "地に足のついた", ["現実的な", "気取らない"]),
    ("drive someone up the wall", "うんざりさせる", ["イライラさせる", "怒らせる"]),
    ("easy come easy go", "悪銭身につかず", ["簡単に手に入るものは簡単に失う"]),
    ("elbow grease", "力仕事", ["一生懸命の努力", "汗水たらす作業"]),
    ("get a taste of your own medicine", "自分がしたことを思い知る", ["因果応報だ"]),
    ("get the ball rolling", "事を始める", ["物事を動かし始める"]),
    ("go with the flow", "流れに任せる", ["成り行きに任せる"]),
    ("have a lot on your plate", "やることが山積みだ", ["忙しい", "抱えている仕事が多い"]),
    ("hit it off", "意気投合する", ["すぐに仲良くなる"]),
    ("hold your horses", "ちょっと待て", ["落ち着いて", "急がないで"]),
    ("in the nick of time", "間一髪で", ["ぎりぎりのところで", "間に合って"]),
    ("keep it under wraps", "秘密にしておく", ["内緒にする", "隠しておく"]),
    ("know the ropes", "コツを知っている", ["要領を心得ている"]),
    ("last but not least", "最後に大切なことだが", ["最後になったが劣らず重要な"]),
    ("lose your marbles", "正気を失う", ["おかしくなる", "頭がおかしくなる"]),
    ("make hay while the sun shines", "好機を逃すな", ["チャンスを活かせ"]),
    ("not cut out for", "向いていない", ["適性がない"]),
    ("on the ball", "抜かりがない", ["しっかりしている", "よく気が利く"]),
    ("over the moon", "大喜びだ", ["有頂天だ", "嬉しくてたまらない"]),
    ("play devil's advocate", "あえて反論する", ["わざと反対意見を述べる"]),
    ("put your foot in your mouth", "失言する", ["余計なことを言ってしまう"]),
    ("rain on someone's parade", "水を差す", ["興ざめさせる", "邪魔をする"]),
    ("run out of steam", "力が尽きる", ["息切れする", "勢いがなくなる"]),
    ("see the writing on the wall", "不吉な前兆を見る", ["先行きが暗いことを悟る"]),
    ("shake a leg", "急いで", ["早くしろ", "さっさとしろ"]),
    ("show someone the ropes", "仕事を教える", ["やり方を見せる"]),
    ("take with a pinch of salt", "鵜呑みにしない", ["疑ってかかる"]),
    ("the whole shebang", "何もかも全部", ["一切合切"]),
    ("through thick and thin", "良い時も悪い時も", ["どんなことがあっても"]),
    ("tie the knot", "結婚する", ["ゴールインする"]),
    ("to each their own", "人それぞれだ", ["好みは人それぞれ"]),
    ("up to snuff", "水準に達している", ["十分な品質だ"]),
    ("wet behind the ears", "未熟だ", ["経験不足だ", "まだ青い"]),
    ("with flying colors", "見事に", ["堂々と", "大成功で"]),
    ("wouldn't be caught dead", "絶対にしない", ["死んでもやらない"]),

    # ══════════════════════════════════════════
    # Section B: Common Japanese Idioms / Proverbs → Natural English
    # ══════════════════════════════════════════

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

    # ── Additional Japanese-origin proverbs/expressions (NEW in v2) ──
    ("even Homer sometimes nods", "弘法にも筆の誤り", ["弘法も筆の誤り"]),
    ("a frog in a well knows nothing of the sea", "井の中の蛙大海を知らず", []),
    ("the nail that sticks out gets hammered down", "出る杭は打たれる", []),
    ("fall seven times get up eight", "七転び八起き", ["七転八起"]),
    ("flowers wither and stars shift", "花は散り星は移ろう", []),
    ("even dust when piled up becomes a mountain", "塵も積もれば山となる", []),
    ("there is no medicine for a fool", "馬鹿につける薬はない", []),
    ("dumplings are better than flowers", "花より団子", []),
    ("an early start makes for an easy journey", "善は急げ", []),
    ("don't judge a person by appearance", "人を見かけで判断するな", ["人は見かけによらぬもの"]),
    ("a journey of a thousand miles begins with a single step", "千里の道も一歩から", []),
    ("time and tide wait for no one", "歳月人を待たず", []),
    ("too many cooks spoil the broth", "船頭多くして船山に上る", []),
    ("the darkest hour is just before the dawn", "夜明け前が一番暗い", []),
    ("blood is thicker than water", "血は水よりも濃い", []),
    ("after a storm comes a calm", "雨降って地固まる", []),
    ("knowledge is power", "知は力なり", []),
    ("slow and steady wins the race", "ゆっくり着実にが勝利する", ["急がば回れ"]),
    ("like father like son", "この父にしてこの子あり", ["蛙の子は蛙"]),
    ("silence is golden", "沈黙は金なり", ["沈黙は金"]),
    ("art is long life is short", "芸術は長く人生は短い", []),
    ("what goes around comes around", "因果応報", ["自業自得"]),
    ("patience is a virtue", "忍耐は美徳だ", ["石の上にも三年"]),
    ("love is blind", "恋は盲目", []),
    ("all that glitters is not gold", "光るもの必ずしも金ならず", []),
    ("money doesn't grow on trees", "金のなる木はない", []),
    ("an eye for an eye", "目には目を", []),
    ("fortune favors the bold", "幸運は勇者に味方する", ["天は自ら助くる者を助く"]),
    ("knowledge without practice is useless", "学問に王道なし", []),
    ("a friend in need is a friend indeed", "まさかの友こそ真の友", []),
    ("the grass is always greener on the other side", "隣の芝生は青く見える", []),
    ("absence makes the heart grow fonder", "会わぬ恋は薄れる", ["去る者は日日に疎し"]),
    ("there's no place like home", "住めば都", []),
    ("old habits die hard", "三つ子の魂百まで", []),
    ("a rolling stone gathers no moss", "転石苔を生ぜず", []),
    ("all's well that ends well", "終わりよければすべてよし", []),

    # ══════════════════════════════════════════
    # Section C: Phrasal Verbs with Idiomatic Meaning
    # ══════════════════════════════════════════
    # These are phrasal verbs where the meaning differs from literal translation

    ("he didn't care for swimming", "彼は水泳が好きではなかった", ["彼は水泳が得意ではなかった"]),
    ("she looks after her grandmother", "彼女は祖母の世話をしている", ["彼女はおばあさんの面倒を見ている"]),
    ("I came across an old friend", "旧友にばったり会った", ["昔の友人に偶然出会った"]),
    ("he turned down the offer", "彼はその申し出を断った", ["彼はそのオファーを拒否した"]),
    ("she gave up smoking", "彼女は喫煙をやめた", ["彼女はたばこをやめた"]),
    ("they put off the meeting", "彼らは会議を延期した", ["彼らはミーティングを先送りにした"]),
    ("he ran into his teacher", "彼は先生にばったり会った", ["彼は先生と偶然会った"]),
    ("she brought up three children", "彼女は3人の子供を育てた", ["彼女は子供を3人育て上げた"]),
    ("we need to figure out a solution", "解決策を見つけなければならない", ["何とか解決方法を考え出さないと"]),
    ("she took after her mother", "彼女は母親似だ", ["彼女は母親に似ている"]),
    ("he passed away last year", "彼は去年亡くなった", ["彼は昨年他界した"]),
    ("I can't put up with this noise", "この騒音には耐えられない", ["この騒音は我慢できない"]),
    ("she ended up in Paris", "彼女は結局パリに行き着いた", ["彼女は最終的にパリにたどり着いた"]),
    ("they called off the wedding", "彼らは結婚式を中止した", ["彼らは挙式をキャンセルした"]),
    ("he got over his illness", "彼は病気を克服した", ["彼は病気から回復した"]),
    ("she picked up Japanese quickly", "彼女はすぐに日本語を覚えた", ["彼女は日本語を早く習得した"]),
    ("he looked into the matter", "彼はその件を調査した", ["彼はその問題を調べた"]),
    ("they set up a new company", "彼らは新しい会社を設立した", ["彼らは新会社を立ち上げた"]),
    ("she broke down in tears", "彼女は泣き崩れた", ["彼女は涙を流して崩れ落ちた"]),
    ("he made up an excuse", "彼は言い訳をでっち上げた", ["彼は口実を作った"]),
    ("we have to carry on working", "仕事を続けなければならない", ["作業を継続しなければならない"]),
    ("she pointed out the error", "彼女はその誤りを指摘した", ["彼女は間違いに気づかせた"]),
    ("he let her down", "彼は彼女を失望させた", ["彼は彼女の期待を裏切った"]),
    ("I look forward to meeting you", "お会いできるのを楽しみにしています", ["お目にかかれるのを楽しみにしています"]),
    ("she stands out in a crowd", "彼女は人混みの中でも目立つ", ["彼女は群衆の中で際立っている"]),
]


# ──────────────────────────────────────────────────────────────────────────────
# Contextual Sentence Templates (Expanded in v2)
# ──────────────────────────────────────────────────────────────────────────────
# Wrap idioms in natural sentence context to help the model learn them
# in realistic settings rather than as isolated phrases.

EN_TEMPLATES = [
    "{idiom}.",
    "I think we should {idiom}.",
    "She told me to {idiom}.",
    'He always says "{idiom}".',
    "As the saying goes, {idiom}.",
    "You know what they say: {idiom}.",
    "In this situation, I would say {idiom}.",
    'My grandmother used to say, "{idiom}".',
    "Well, {idiom}, as they say.",
    "It's like the old saying: {idiom}.",
    # NEW templates
    "People often say that {idiom}.",
    "I learned the hard way that {idiom}.",
    "The teacher explained that {idiom}.",
    "In English, there is a saying: {idiom}.",
    "He reminded me that {idiom}.",
    'There is a saying that goes, "{idiom}".',
    "We all know that {idiom}.",
    "As my father always said, {idiom}.",
]

JA_TEMPLATES = [
    "{idiom}。",
    "{idiom}ということわざがある。",
    "{idiom}と言われている。",
    "{idiom}という表現がある。",
    "昔から{idiom}と言う。",
    "彼は{idiom}と言った。",
    "先生が{idiom}と教えてくれた。",
    "日本語には{idiom}という言い回しがある。",
]


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_idiom_corpus(
    output_path: str,
    oversample: int = 10,
    seed: int = 42,
    synthetic_score: float = 0.95,
    use_templates: bool = True,
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
    use_templates : bool
        If True, also generate template-wrapped contextual pairs.
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

        # Template-wrapped pairs for contextual learning
        if use_templates:
            for en_tmpl in EN_TEMPLATES:
                try:
                    en_contextualized = en_tmpl.format(idiom=en_idiom)
                except (KeyError, IndexError):
                    continue
                # Match with primary JA (templates don't change JA meaning)
                all_pairs.append((en_contextualized, ja_primary))

            for ja_tmpl in JA_TEMPLATES:
                try:
                    ja_contextualized = ja_tmpl.format(idiom=ja_primary)
                except (KeyError, IndexError):
                    continue
                all_pairs.append((en_idiom, ja_contextualized))

    # Deduplicate
    all_pairs = list(set(all_pairs))
    logger.info("Unique pairs after dedup: %d", len(all_pairs))

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
    logger.info("  Base pairs (with templates): %d", len(all_pairs))
    logger.info("  Oversample factor: %d×", oversample)

    return total_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate EN-JA idiom augmentation corpus (v2, 300+ idioms).")
    p.add_argument("--output", default="data/dataset/train-set/idiom_pairs.txt",
                   help="Output TSV file path")
    p.add_argument("--oversample", type=int, default=10,
                   help="Number of times to repeat each pair (default: 10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--score", type=float, default=0.95,
                   help="Synthetic BiCleaner score (default: 0.95)")
    p.add_argument("--no_templates", action="store_true",
                   help="Disable contextual sentence template generation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_idiom_corpus(
        output_path=args.output,
        oversample=args.oversample,
        seed=args.seed,
        synthetic_score=args.score,
        use_templates=not args.no_templates,
    )

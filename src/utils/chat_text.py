from collections import Counter
from collections.abc import Mapping
from typing import List

from models import Message, Reaction
from whatsapp.jid import parse_jid


# If no reactions, don't display anything.
# Group Reactions by count per reaction. E.g: "👍 5, 👎 2"
# If all reactions has only one count, don't need to display number. E.G: "👍, 👎, 😁"
def render_reactions(reactions: List[Reaction]) -> str:
    if not reactions:
        return ""

    counts = Counter(r.emoji for r in reactions)
    if all(count == 1 for count in counts.values()):
        return f"Reactions: {', '.join(counts.keys())}"

    return (
        f"Reactions: {', '.join(f'{emoji} {count}' for emoji, count in counts.items())}"
    )


def chat2text(
    history: List[Message],
    opt_out_map: dict[str, str],
    trusted_names: Mapping[str, str] | None = None,
) -> str:
    lines = []
    for message in history:
        sender_jid = parse_jid(message.sender_jid)
        sender_user = sender_jid.user
        normalized_sender = message.sender_jid
        if sender_user in opt_out_map:
            sender_display = opt_out_map[sender_user]
        elif trusted_names and normalized_sender in trusted_names:
            sender_display = trusted_names[normalized_sender]
        else:
            sender_display = f"@{sender_user}"

        reaction_text = render_reactions(message.reactions)
        if reaction_text:
            lines.append(
                f"{message.timestamp}: {sender_display}: {message.text}. {reaction_text}"
            )
        else:
            lines.append(f"{message.timestamp}: {sender_display}: {message.text}")

    return "\n".join(lines)

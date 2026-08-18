"""Which platform env vars the setup surfaces hide.

Every messaging platform ships the same handful of knobs that are either set
for the user later or already correct by default. Listing them on a setup form
turns "paste your bot token" into a five-field interrogation where many
answers are not discoverable during first-run setup.

Hiding them is a *presentation* decision only. The env vars keep working
through ``hermes config set``, ``.env``, and ``config.yaml``; the gateway reads
them exactly as before. This module just says what a new user is asked during
setup.

Lives in a small dependency-free module so setup callers can share it.
"""

# Suffix matching keeps the retained adapters consistent without duplicating
# each environment-variable spelling.
#
#   *_HOME_CHANNEL*        the bot offers /sethome on the first chat
#   *_ALLOW_ALL_USERS      defaults off; enabling it is a security decision
#   *_REPLY_TO_MODE        cosmetic threading preference
#   *_REPLY_MODE           same, Mattermost's spelling
#   *_REQUIRE_MENTION      behavior toggle with a sane default
#   *_AUTO_THREAD          same
#   *_FREE_RESPONSE_*      per-channel tuning, done once the bot is in a server
#   *_ALLOWED_CHANNELS     same
#   *_PROXY                only for networks that block the platform
#
# Allowlists (*_ALLOWED_USERS) deliberately stay visible: that IS the decision
# a new user has to make, and the gateway denies everyone until it's set.
SETUP_HIDDEN_ENV_SUFFIXES = (
    "_HOME_CHANNEL",
    "_HOME_CHANNEL_NAME",
    "_HOME_CHANNEL_THREAD_ID",
    "_HOME_ADDRESS",
    "_ALLOW_ALL_USERS",
    "_REPLY_TO_MODE",
    "_REPLY_MODE",
    "_REQUIRE_MENTION",
    "_AUTO_THREAD",
    "_FREE_RESPONSE_CHANNELS",
    "_FREE_RESPONSE_ROOMS",
    "_ALLOWED_CHANNELS",
    "_PROXY",
)


def is_setup_hidden_env(name: str) -> bool:
    """True when a var is self-configuring and shouldn't appear in setup forms.

    Callers must still keep any var a platform lists as *required* — hiding a
    required credential would make that platform unconfigurable from the UI.
    """
    return name.endswith(SETUP_HIDDEN_ENV_SUFFIXES)

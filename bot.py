from __future__ import annotations

import asyncio
import atexit
import io
import json
import logging
import os
import sys
from contextlib import suppress
from enum import IntEnum
from functools import total_ordering
from graphlib import CycleError, TopologicalSorter
from typing import Any, Final, cast

import discord
from attrs import Attribute, asdict, define, field, frozen
from discord import abc
from discord.ext import commands  # type: ignore[attr-defined]
from dotenv import load_dotenv

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
# --------

# Constants
load_dotenv()

TOKEN: Final[str] = cast(str, os.getenv("TOKEN"))

INTENTS: Final = discord.Intents(
    guilds=True,
    emojis_and_stickers=False,
    members=False,
    bans=False,
    integrations=False,
    webhooks=False,
    invites=False,
    voice_states=False,
    presences=False,
    messages=False,
    reactions=False,
    typing=False,
)
# ----------


def manage_channels_check(ctx: discord.ApplicationContext) -> bool:
    return cast(bool, commands.has_guild_permissions(manage_channels=True).predicate(ctx))


def guild_state_serializer(inst: type, field: Attribute, value: Any) -> Any:  # type: ignore[type-arg]
    if isinstance(value, abc.GuildChannel):
        return value.name
    elif isinstance(value, Timer):
        return value.active()
    else:
        return value


class Priority(IntEnum):
    TextChannel = 0
    VoiceChannel = 1
    CategoryChannel = 2
    StageChannel = 3

    @classmethod
    def from_abc(cls, channel: abc.GuildChannel):
        return cls[channel.__class__.__name__]


class Timer:
    def __init__(self, time: int | float = 0):
        self._task = self.new(time)

    def active(self) -> bool:
        return not self._task.done()

    def reset(self, time: int | float) -> None:
        self._task = self.new(time)

    @staticmethod
    def new(time: int | float):
        return asyncio.create_task(asyncio.sleep(time))


@frozen
class ChannelDiff:
    from_pos: int
    to_pos: int
    channel: abc.GuildChannel

    def move(self):
        return self.to_pos - self.from_pos


@frozen
@total_ordering
class ChannelState:
    position: int
    channel: abc.GuildChannel
    category: discord.CategoryChannel | None

    @classmethod
    def from_channel(cls, channel: abc.GuildChannel):
        return cls(channel.position, channel, channel.category)

    @property
    def category_pos(self) -> int:
        if self.category is None:
            return -1
        else:
            return cast(int, self.category.position)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelState):
            return NotImplemented
        return (self.position, self.category) == (other.position, other.category)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ChannelState):
            return NotImplemented

        self_priority, other_priority = (
            Priority.from_abc(self.channel),
            Priority.from_abc(other.channel),
        )

        if self.category_pos == other.category_pos:
            if self_priority > other_priority:
                return False
            elif self_priority == other_priority:
                return self.position <= other.position
            else:
                return True
        elif self.category_pos < other.category_pos:
            if self_priority == 2:
                if self.position > other.category_pos:
                    return False
            return True
        else:
            if self.category_pos < other.position and 2 in (self_priority, other_priority):
                return True
            else:
                return False


@define
class GuildState:
    channels: dict[int, ChannelState]
    timer: Timer = field(factory=Timer, init=False)
    ongoing: list[ChannelDiff] = field(factory=list, init=False)
    ongoing_cat: dict[discord.CategoryChannel, int] = field(factory=dict, init=False)
    creation: list[abc.GuildChannel] = field(factory=list, init=False)

    @classmethod
    def from_guild(cls, guild: discord.Guild) -> GuildState:
        return cls({c.id: ChannelState.from_channel(c) for c in guild.channels})

    def update(self, guild: discord.Guild) -> None:
        self.channels = {c.id: ChannelState.from_channel(c) for c in guild.channels}

    def asdict(self):
        return asdict(
            self,
            filter=lambda attr, _: attr.name != "channels",
            value_serializer=guild_state_serializer,
        ) | {
            "channels": [
                asdict(cs, value_serializer=guild_state_serializer)
                for cs in sorted(self.channels.values())
            ]
        }


class Bot(discord.Bot):

    configs: dict[int, int] = {}
    guilds_state: dict[int, GuildState] = {}

    def __init__(self):
        super().__init__(intents=INTENTS)
        atexit.register(self.save)

    async def channel_log(self, guild: discord.Guild, message) -> None:
        with suppress(KeyError, AttributeError):
            await guild.get_channel(self.configs[guild.id]).send(message)

    async def on_ready(self):
        logger.info((t := f"Logged in as {self.user} (ID: {self.user.id})"))
        logger.info(len(t) * "-")

        for guild in self.guilds:
            self.guilds_state[guild.id] = GuildState.from_guild(guild)

        with suppress(FileNotFoundError):
            with open("db.json", "r") as db:
                self.configs = {int(guild_id): ch_id for guild_id, ch_id in json.load(db).items()}
                logger.info("Loaded guild configs into memory")

    async def on_guild_join(self, guild: discord.Guild):
        self.guilds_state[guild.id] = GuildState.from_guild(guild)

    async def on_guild_leave(self, guild: discord.Guild):
        self.guilds_state.pop(guild.id, None)

    async def on_guild_channel_delete(self, channel: abc.GuildChannel):
        guild: discord.Guild = channel.guild
        self.guilds_state[guild.id].update(guild)

    async def on_guild_channel_create(self, channel: abc.GuildChannel):
        guild: discord.Guild = channel.guild
        state = self.guilds_state[guild.id]
        state.update(guild)

        log = await channel.guild.audit_logs(
            limit=1, action=discord.AuditLogAction.channel_create
        ).next()
        creator: discord.abc.User = log.user
        state.creation.append(channel)

        def callback(_):
            with suppress(ValueError):
                state.creation.remove(channel)

        asyncio.create_task(asyncio.sleep(2 * 60)).add_done_callback(callback)

        if not creator.bot:
            await creator.send(
                "You have just created a channel, you may move it freely "
                "for the next 2 minutes. To move other channels `/unlock`"
            )

    async def on_guild_channel_update(  # noqa: C901
        self, before: abc.GuildChannel, after: abc.GuildChannel
    ):
        guild: discord.Guild = after.guild
        if before.position != after.position:
            # Channel chnaged position
            state: GuildState = self.guilds_state[guild.id]
            target_pos: int = state.channels[after.id].position

            # When a channel is moved up by the user, that will be the first event,
            # all other channels will move down to accommodate the change, starting from the top.

            # When a channel is moved down by the user, that will be the last event,
            # all other channels will move up to accommodate the change, starting from the top.

            diff = ChannelDiff(before.position, after.position, after)

            if isinstance(after, discord.CategoryChannel):
                state.ongoing_cat[after] = len(after.channels)
            else:
                category = after.category
                try:
                    value = state.ongoing_cat.pop(category)
                except KeyError:
                    pass
                else:
                    # This channel is being moved because its category is being moved
                    if value == 1:
                        # This was the last channel to move due to a category move
                        # Don't reasing
                        pass
                    else:
                        state.ongoing_cat[category] = value
                    return

            state.ongoing.append(diff)

            if diff.move() < -1:
                # First position change when moving up by more then one

                # Do channels need to be moved to accomodate?
                if next(
                    filter(lambda cs: cs.position == after.position, state.channels.values()), None
                ):  # Yes
                    #  Wait until full cycle
                    pass
                else:  # No
                    state.ongoing.clear()

                    if after.position != target_pos:
                        if state.timer.active() or after in state.creation:
                            state.update(guild)
                        else:
                            logger.info(
                                f"Bot is moving {after.name} back to {target_pos} on {guild.name}"
                            )
                            await after.edit(position=target_pos)
                    else:
                        logger.info(f"Moved {after.name} back to {target_pos} on {guild.name}")
                        await self.channel_log(guild, "Someone attemped to move a channel")

            elif diff.move() == -1:
                # First step in moving a channel by one, either up or down
                # Will never be a last step
                pass

            elif diff.move() > 1:
                # Final position change when moving down by more then one
                state.ongoing.clear()

                if after.position != target_pos:
                    if state.timer.active() or after in state.creation:
                        state.update(guild)
                    else:
                        logger.info(
                            f"Bot is moving {after.name} back to {target_pos} on {guild.name}"
                        )
                        await after.edit(position=target_pos)
                else:
                    logger.info(f"Moved {after.name} back to {target_pos} on {guild.name}")
                    await self.channel_log(guild, "Someone attemped to move a channel")

            else:
                # Second step in moving a channel by one, either up or down
                # Could be a last step

                try:
                    TopologicalSorter({cd.to_pos: {cd.from_pos} for cd in state.ongoing}).prepare()
                except CycleError:
                    # Cycle has been completed
                    original = next(filter(lambda cd: cd.from_pos == after.position, state.ongoing))

                    state.ongoing.clear()

                    if after.position != target_pos:
                        if state.timer.active() or original.channel in state.creation:
                            state.update(guild)
                        else:
                            logger.info(
                                f"Bot is moving {original.channel.name} back to {original.from_pos} on {guild.name}"
                            )
                            await original.channel.edit(position=original.from_pos + 1)
                    else:
                        logger.info(f"Moved {after.name} back to {target_pos} on {guild.name}")
                        await self.channel_log(guild, "Someone attemped to move a channel")
                else:
                    # Cycle has not yet been completed
                    pass

    async def on_application_command_error(self, context: discord.ApplicationContext, exception):
        if isinstance(exception, discord.commands.errors.ApplicationCommandInvokeError):
            exception = exception.original
        if isinstance(exception, commands.errors.MissingPermissions):
            await context.respond(exception.args[0], ephemeral=True)
        else:
            await super().on_application_command_error(context, exception)

    def save(self):
        with open("db.json", "w") as db:
            json.dump(self.configs, db)
            logger.info("Saved configs to persistent")


bot: Final = Bot()


@bot.slash_command(
    name="unlock",
    description="Unlocks moving channels",
    checks=[manage_channels_check],
    options=[
        discord.Option(
            discord.enums.SlashCommandOptionType.integer,
            name="time",
            description="Time to leave channels unlocked for, given in minutes",
            required=False,
            min_value=1,
            max_value=5,
            default=5,
        )
    ],
)
async def unlock(ctx: discord.ApplicationContext, time: int):
    inter: discord.Interaction = ctx.interaction
    if inter.guild:
        bot.guilds_state[inter.guild.id].timer.reset(time * 60)
        logger.info(
            f"{inter.user.name} unloked channels on {inter.guild.name} for {time} minute(s)"
        )
        await ctx.respond(f"Channels unlocked for **{time}** minutes")
    else:
        await ctx.respond("Command must be used from within a guild")


admin = bot.create_group("admin", "Management commands")


@admin.command(
    name="set",
    description="Set options for this server",
    checks=[manage_channels_check],
    options=[
        discord.Option(
            discord.enums.SlashCommandOptionType.channel,
            name="logs_channel",
            description="The channel the bot uses for logging",
            channel_types=[discord.enums.ChannelType.text],
            required=False,
        )
    ],
)
async def set_config(ctx: discord.ApplicationContext, logs_channel: discord.TextChannel):
    guild_id: int = ctx.interaction.guild_id
    response = []

    if logs_channel is not None:
        if bot.configs.get(guild_id, None) == logs_channel.id:
            response.append(f"Set logs_channel was already set to {logs_channel.mention} ❓")
        else:
            bot.configs[guild_id] = logs_channel.id
            response.append(f"Set logs_channel to {logs_channel.mention} ✔️")

    if not response:
        response.append("Nothing changed as no option was set ❌")

    await ctx.respond("\n".join(response), ephemeral=True)


@bot.slash_command(
    name="debug",
    description="Debugging command",
    options=[
        discord.Option(
            discord.enums.SlashCommandOptionType.string,
            name="cmd",
            description="cmd to execute",
            required=True,
            autocomplete=discord.utils.basic_autocomplete(
                ["guildstate", "shutdown", "guilds", "reset"]
            ),
        ),
        discord.Option(
            discord.enums.SlashCommandOptionType.string,
            name="args",
            description="args to pass",
            required=False,
        ),
    ],
)
@discord.commands.permissions.is_owner()
async def debug(ctx: discord.ApplicationContext, cmd: str, args: str = ""):
    guild: discord.Guild | None = ctx.interaction.guild
    if cmd == "guilds":
        await ctx.respond("\n".join(map(lambda g: cast(str, g.name), bot.guilds)))
    elif cmd == "guildstate":
        if guild:
            with io.StringIO(
                json.dumps(
                    bot.guilds_state[guild.id].asdict(),
                    indent=2,
                )
            ) as f:
                await ctx.respond(file=discord.File(f, "guildstate.json"))
        else:
            await ctx.respond("ERROR: Not in a guild")
    elif cmd == "shutdown":
        await ctx.respond("Shutting down")
        sys.exit()
    elif cmd == "reset":
        if guild:
            bot.guilds_state[guild.id] = GuildState.from_guild(guild)
    else:
        await ctx.respond("Invalid command")


bot.run(TOKEN)

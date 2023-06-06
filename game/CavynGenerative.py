import os
import sys
import random
import math
import numpy as np
import music21 as m21
import copy
import sys
from queue import PriorityQueue
import pickle
from concurrent.futures import ThreadPoolExecutor

import pygame
from pygame.locals import *
import pygame.midi

from data.scripts.entity import Entity
from data.scripts.anim_loader import AnimationManager
from data.scripts.text import Font

executor = ThreadPoolExecutor(max_workers=4)

class Measure:
    def __init__(self, measure, instrument):
        self.measure = measure
        self.instrument = instrument
    
    def __repr__(self):
        return f'Measure of {self.instrument}'


class GenerativeMusic:
    def __init__(self) -> None:
        self.instruments = ['Str', 'Hp', 'Hn', None, 'Timp', 'Ch']
        self.instrument_association = {
            'Str': m21.instrument.StringInstrument(),
            'Hp': m21.instrument.Harp(),
            'Hn': m21.instrument.Horn(),
            'Timp': m21.instrument.Timpani(),
            'Ch': m21.instrument.Choir(),
            # None is for the case where the instrument is not specified
            # and will use any kind of percussion instrument
            None: m21.instrument.Percussion()
        }
        self.transition_matrix = np.loadtxt('transition_matrix.csv', delimiter=',')
        with open('measures_list.pkl', 'rb') as f:
            self.measures_list = pickle.load(f)
    
class MidiPlayer:
    def __init__(self) -> None:
        self.music_out_priority_queue = PriorityQueue()
        pygame.midi.init()
        self.port = pygame.midi.get_default_output_id()
        print(f'\n\nDefault output port: {self.port}')
        self.midi_out = pygame.midi.Output(self.port, latency=50, buffer_size=1024)
        self.g_timestamp = pygame.midi.time()
        self.generative_music = GenerativeMusic()

        instrument_map_ = {1: 48, 6: 46, 8: 60, 11: 47, 13: 52, 2: None}

        self.global_instrument_map = {}

        for channel, instrument in instrument_map_.items():
            if instrument == None:
                self.midi_out.set_instrument(49, 9)
                self.global_instrument_map[instrument] = 9
            else:
                self.midi_out.set_instrument(instrument, channel - 1)
                self.global_instrument_map[instrument] = channel - 1
                
        self.midi_out.set_instrument(88, 3)
        self.midi_out.set_instrument(54, 4)
        self.midi_out.set_instrument(102, 5)

        self.is_jump_note_on = False
        self.is_warp_note_on = False
        self.is_cube_note_on = False
    
        init_midi_tracks, init_inv_map, self.current_idx = self.initialize_music()

        events = self.convert_to_pygame_events(init_midi_tracks, init_inv_map, init=True)
        for event in events:
            self.music_out_priority_queue.put_nowait(event)
    
        pass
    
    
    def initialize_music(self):
        song = m21.stream.Score()
        parts = [m21.stream.Part() for _ in range(len(self.generative_music.instrument_association))]
        for index, instrument in enumerate(self.generative_music.instruments):
            song.insert(0, parts[index])

         # get the number of measures in the song
        measure_num = len(song.parts[0].getElementsByClass(m21.stream.Measure))
        # use the transition matrix to get the next measure
        current_idx = 0
        measures_in_order = self.generative_music.measures_list[current_idx]
        parts = song.parts
        for index, measure in enumerate(measures_in_order):
            if measure is None:
                continue
            if index == 0 and measure_num == 0:
                parts[index].insert(0, self.generative_music.instrument_association[measure.instrument])
            newMeasure = copy.deepcopy(measure.measure)
            parts[index].insert(measure_num * 4, newMeasure)

        song = m21.midi.translate.prepareStreamForMidi(song)
        tup, _ = m21.midi.translate.channelInstrumentData(song)
        # get all midi events from the song
        song.makeMeasures(inPlace=True)
        self.song = song
        midi_tracks = m21.midi.translate.streamHierarchyToMidiTracks(song)
        inv_map = {v: k for k, v in tup.items()}
        return  midi_tracks, inv_map, current_idx

    def handle_music_event(self, event, current_item, is_dead):
        pitch, velocity, channel = event[1:]
        if current_item == 'jump':
            pitch = min(pitch + 16, 127)
        elif current_item == 'warp':
            pitch = max(pitch - 16, 0)
        elif current_item == 'cube':
            velocity = min(velocity + 20, 127)
        if is_dead and event[0]:
            return
        if event[0]:
            self.midi_out.note_on(pitch, velocity // 2, channel)
        else:
            self.midi_out.note_off(pitch, velocity // 2, channel)
        return
    
    # convert the midi events to events that can be played by pygame
    # of the form ([status, data1=0, data2=0, ...], timestamp)
    # where status is a midi event type in hex
    def convert_to_pygame_events(self, midi_tracks_, instrument_channels, init=False, bpm = 50, tpb = 2048):
        events = []
        new_max_timestamp = 0
        bpms = bpm / 60000
        milli_per_beat = 1 / bpms
        milli_per_tick = milli_per_beat / tpb
        new_max_timestamp = 0
        for track in midi_tracks_:
            timestamp = self.g_timestamp
            last_note_timestamp = 0
            for midi_event in track.events:
                if midi_event.isDeltaTime():
                    timestamp += midi_event.time * milli_per_tick
                if midi_event.isNoteOn() or midi_event.isNoteOff():
                    instrument = instrument_channels[midi_event.channel] if midi_event.channel != 10 else None
                    events.append((timestamp, [midi_event.isNoteOn(), midi_event.parameter1, midi_event.parameter2, self.global_instrument_map[instrument]]))
                    last_note_timestamp = timestamp
            new_max_timestamp = max(new_max_timestamp, last_note_timestamp)
        self.g_timestamp = new_max_timestamp
        return events
    
    def gen_new_music(self, num_coins):
        new_song = m21.stream.Score()
        new_parts = [m21.stream.Part() for _ in range(len(self.generative_music.instrument_association))]
        for index, instrument in enumerate(self.generative_music.instruments):
            new_song.insert(0, new_parts[index])
        
        # get the number of measures in the song
        measure_num = len(new_song.parts[0].getElementsByClass(m21.stream.Measure))
        # use the transition matrix to get the next measure
        self.current_idx = np.random.choice(len(self.generative_music.transition_matrix[self.current_idx]), p=self.generative_music.transition_matrix[self.current_idx])
        measures_in_order = self.generative_music.measures_list[self.current_idx]
        parts = new_song.parts
        for index, measure in enumerate(measures_in_order):
            if measure is None:
                continue
            if index == 0 and measure_num == 0:
                parts[index].insert(0, self.generative_music.instrument_association[measure.instrument])
            newMeasure = copy.deepcopy(measure.measure)
            parts[index].insert(measure_num * 4, newMeasure)
    
        new_song.makeMeasures(inPlace=True)
        new_song = m21.midi.translate.prepareStreamForMidi(new_song)
        new_song.makeMeasures(inPlace=True)
        new_tup, _ = m21.midi.translate.channelInstrumentData(new_song)
        # get all midi events from the song
        new_song.makeMeasures(inPlace=True)
        new_midi_tracks = m21.midi.translate.streamHierarchyToMidiTracks(new_song)
        new_inv_map = {v: k for k, v in new_tup.items()}
        bpm = 50 + num_coins
        new_events = self.convert_to_pygame_events(new_midi_tracks, new_inv_map, bpm=min(bpm, 200))
        for event in new_events:
            self.music_out_priority_queue.put_nowait(event)

    
    def handle_music_events(self, current_item, is_dead):
        if self.is_jump_note_on and current_item != 'jump':
            self.midi_out.note_off(70, 0, 3)
            self.is_jump_note_on = False
        
        elif self.is_warp_note_on and current_item != 'warp':
            self.midi_out.note_off(70, 0, 4)
            self.is_warp_note_on = False
        
        elif self.is_cube_note_on and current_item != 'cube':
            self.midi_out.note_off(50, 0, 5)
            self.is_cube_note_on = False
        
        if current_item == 'jump':
            if not self.is_jump_note_on:
                self.midi_out.note_on(70, 127, 3)
                self.is_jump_note_on = True
        elif current_item == 'warp':
            if not self.is_warp_note_on:
                self.midi_out.note_on(70, 127, 4)
                self.is_warp_note_on = True
        elif current_item == 'cube':
            if not self.is_cube_note_on:
                self.midi_out.note_on(50, 127, 5)
                self.is_cube_note_on = True
                
        while pygame.midi.time() >= self.music_out_priority_queue.queue[0][0]:
            event = self.music_out_priority_queue.get_nowait()[1]
            self.handle_music_event(event, current_item, is_dead)
        
            
class Item(Entity):
    def __init__(self, *args, velocity=[0, 0]):
        super().__init__(*args)
        self.velocity = velocity
        self.time = 0

    def update(self, tiles):
        self.time += 1
        self.velocity[1] = min(self.velocity[1] + 0.2, 3)
        self.velocity[0] = self.game.normalize(self.velocity[0], 0.05)
        self.move(self.velocity, tiles)

class Player(Entity):
    def __init__(self, *args):
        super().__init__(*args)
        self.velocity = [0, 0]
        self.right = False
        self.left = False
        self.speed = 1.4
        self.jumps = 2
        self.jumps_max = 2
        self.jumping = False
        self.jump_rot = 0
        self.air_time = 0

    def attempt_jump(self):
        if self.jumps:
            if self.jumps == self.jumps_max:
                self.velocity[1] = -5
            else:
                self.velocity[1] = -4
                for i in range(24):
                    physics_on = random.choice([False, False, True])
                    direction = 1
                    if i % 2:
                        direction = -1
                    self.game.sparks.append([[self.center[0] + random.random() * 14 - 7, self.center[1]], [direction * (random.random() * 0.05 + 0.05) + (random.random() * 4 - 2) * physics_on, random.random() * 0.05 + random.random() * 2 * physics_on], random.random() * 3 + 3, 0.04 - 0.02 * physics_on, (6, 4, 1), physics_on, 0.05 * physics_on])
            self.jumps -= 1
            self.jumping = True
            self.jump_rot = 0

    def update(self, tiles):
        super().update(1 / 60)
        self.air_time += 1
        if self.jumping:
            self.jump_rot += 16
            if self.jump_rot >= 360:
                self.jump_rot = 0
                self.jumping = False
            if self.flip[0]:
                self.rotation = -self.jump_rot
            else:
                self.rotation = self.jump_rot
            self.scale[1] = 0.7
        else:
            self.scale[1] = 1

        self.velocity[1] = min(self.velocity[1] + 0.3, 4)
        motion = self.velocity.copy()
        if not self.game.dead:
            if self.right:
                motion[0] += self.speed
            if self.left:
                motion[0] -= self.speed
            if motion[0] > 0:
                self.flip[0] = True
            if motion[0] < 0:
                self.flip[0] = False

        if self.air_time > 3:
            self.set_action('jump')
        elif motion[0] != 0:
            self.set_action('run')
        else:
            self.set_action('idle')

        collisions = self.move(motion, tiles)
        if collisions['bottom']:
            self.jumps = self.jumps_max
            self.jumping = False
            self.rotation = 0
            self.velocity[1] = 0
            self.air_time = 0

      
class Game:
    def __init__(self) -> None:
        self.TILE_SIZE = 16

        self.clock = pygame.time.Clock()
        pygame.init()
        pygame.display.set_caption('Cavyn')
        self.DISPLAY_SIZE = (192, 256)
        self.screen = pygame.display.set_mode((self.DISPLAY_SIZE[0] * 3, self.DISPLAY_SIZE[1] * 3), 0, 32)
        self.display = pygame.Surface(self.DISPLAY_SIZE)
        pygame.mouse.set_visible(False)

        self.WINDOW_TILE_SIZE = (int(self.display.get_width() // 16), int(self.display.get_height() // 16))

        self.tile_img = self.load_img('data/images/tile.png')
        self.chest_img = self.load_img('data/images/chest.png')
        self.ghost_chest_img = self.load_img('data/images/ghost_chest.png')
        self.opened_chest_img = self.load_img('data/images/opened_chest.png')
        self.placed_tile_img = self.load_img('data/images/placed_tile.png')
        self.edge_tile_img = self.load_img('data/images/edge_tile.png')
        self.coin_icon = self.load_img('data/images/coin_icon.png')
        self.item_slot_img = self.load_img('data/images/item_slot.png')
        self.item_slot_flash_img = self.load_img('data/images/item_slot_flash.png')
        self.border_img = self.load_img('data/images/border.png')
        self.border_img_light = self.border_img.copy()
        self.border_img_light.set_alpha(100)
        self.white_font = Font('data/fonts/small_font.png', (251, 245, 239))
        self.black_font = Font('data/fonts/small_font.png', (0, 0, 1))

        self.sounds = {sound.split('/')[-1].split('.')[0] : pygame.mixer.Sound('data/sfx/' + sound) for sound in os.listdir('data/sfx')}
        self.sounds['block_land'].set_volume(0.5)
        self.sounds['coin'].set_volume(0.6)
        self.sounds['chest_open'].set_volume(0.8)
        self.sounds['coin_end'] = pygame.mixer.Sound('data/sfx/coin.wav')
        self.sounds['coin_end'].set_volume(0.35)

        self.item_icons = {
            'cube': self.load_img('data/images/cube_icon.png'),
            'warp': self.load_img('data/images/warp_icon.png'),
            'jump': self.load_img('data/images/jump_icon.png'),
        }

        self.animation_manager = AnimationManager()

        self.GLOW_CACHE = {}

        self.player = Player(self, self.animation_manager, (self.DISPLAY_SIZE[0] // 2 - 5, -20), (8, 16), 'player')
        self.dead = False

        self.tiles = {}
        self.tile_drops = []
        self.bg_particles = []
        self.sparks = []

        self.game_timer = 0
        self.height = 0
        self.target_height = 0
        self.coins = 0
        self.end_coin_count = 0
        self.current_item = None
        self.master_clock = 0
        self.last_place = 0
        self.item_used = False

        for i in range(self.WINDOW_TILE_SIZE[0] - 2):
            self.tiles[(i + 1, self.WINDOW_TILE_SIZE[1] - 1)] = 'tile'

        self.tiles[(1, self.WINDOW_TILE_SIZE[1] - 2)] = 'tile'
        self.tiles[(self.WINDOW_TILE_SIZE[0] - 2, self.WINDOW_TILE_SIZE[1] - 2)] = 'tile'

        self.stack_heights = [self.WINDOW_TILE_SIZE[1] - 1 for i in range(self.WINDOW_TILE_SIZE[0] - 2)]
        self.stack_heights[0] -= 1
        self.stack_heights[-1] -= 1

        self.items = []

        self.counter_fin = 0

        self.midi_player = MidiPlayer()
        self.handling_music = None
        self.submitting_music = None
        pass
    
    def normalize(self, val, amt):
        if val > amt:
            val -= amt
        elif val < -amt:
            val += amt
        else:
            val = 0
        return val

    def lookup_nearby(self, tiles, pos):
        rects = []
        for offset in [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]:
            lookup_pos = (pos[0] // self.TILE_SIZE + offset[0], pos[1] // self.TILE_SIZE + offset[1])
            if lookup_pos in tiles:
                rects.append(pygame.Rect(lookup_pos[0] * self.TILE_SIZE, lookup_pos[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        return rects
    
    def load_img(self, path):
        img = pygame.image.load(path).convert()
        img.set_colorkey((0, 0, 0))
        return img
    
    def glow_img(self, size, color):
        if (size, color) not in self.GLOW_CACHE:
            surf = pygame.Surface((size * 2 + 2, size * 2 + 2))
            pygame.draw.circle(surf, color, (surf.get_width() // 2, surf.get_height() // 2), size)
            surf.set_colorkey((0, 0, 0))
            self.GLOW_CACHE[(size, color)] = surf
        return self.GLOW_CACHE[(size, color)]


    def run(self) -> None:
        while True:
            self.display.fill((22, 19, 40))

            self.master_clock += 1

            parallax = random.random()
            for i in range(2):
                self.bg_particles.append([[random.random() * self.DISPLAY_SIZE[0], self.DISPLAY_SIZE[1] - self.height * parallax], parallax, random.randint(1, 8), random.random() * 1 + 1, random.choice([(0, 0, 0), (22, 19, 40)])])

            for i, p in sorted(enumerate(self.bg_particles), reverse=True):
                size = p[2]
                if p[-1] != (0, 0, 0):
                    size = size * 5 + 4
                p[2] -= 0.01
                p[0][1] -= p[3]
                if size < 1:
                    self.display.set_at((int(p[0][0]), int(p[0][1] + self.height * p[1])), (0, 0, 0))
                else:
                    if p[-1] != (0, 0, 0):
                        pygame.draw.circle(self.display, p[-1], p[0], int(size), 4)
                    else:
                        pygame.draw.circle(self.display, p[-1], p[0], int(size))
                if size < 0:
                    self.bg_particles.pop(i)

            if self.master_clock > 180:
                self.game_timer += 1
                if self.game_timer > 10 + 25 * (20000 - min(20000, self.master_clock)) / 20000:
                    self.game_timer = 0
                    minimum = min(self.stack_heights)
                    options = []
                    for i, stack in enumerate(self.stack_heights):
                        if i != self.last_place:
                            offset = stack - minimum
                            for j in range(int(offset ** (2.2 - min(20000, self.master_clock) / 10000)) + 1):
                                options.append(i)

                    tile_type = 'tile'
                    if random.randint(1, 10) == 1:
                        tile_type = 'chest'
                    c = random.choice(options)
                    self.last_place = c
                    self.tile_drops.append([(c + 1) * self.TILE_SIZE, -self.height - self.TILE_SIZE, tile_type])


            tile_drop_rects = []
            for i, tile in sorted(enumerate(self.tile_drops), reverse=True):
                tile[1] += 1.4
                pos = [tile[0] + self.TILE_SIZE // 2, tile[1] + self.TILE_SIZE]
                r = pygame.Rect(tile[0], tile[1], self.TILE_SIZE, self.TILE_SIZE)
                tile_drop_rects.append(r)

                if r.colliderect(self.player.rect):
                    if not self.dead:
                        self.sounds['death'].play()
                        self.player.velocity = [1.3, -6]
                        for i in range(380):
                            angle = random.random() * math.pi
                            speed = random.random() * 1.5
                            physics_on = random.choice([False, True, True])
                            self.sparks.append([self.player.center.copy(), [math.cos(angle) * speed, math.sin(angle) * speed], random.random() * 3 + 3, 0.02, (12, 2, 2), physics_on, 0.1 * physics_on])
                    self.dead = True

                check_pos = (int(pos[0] // self.TILE_SIZE), int(math.floor(pos[1] / self.TILE_SIZE)))
                if check_pos in self.tiles:
                    self.tile_drops.pop(i)
                    place_pos = (check_pos[0], check_pos[1] - 1)
                    self.stack_heights[place_pos[0] - 1] = place_pos[1]
                    self.tiles[place_pos] = tile[2]
                    self.sounds['block_land'].play()
                    if self.tiles[check_pos] == 'chest':
                        self.tiles[check_pos] = 'tile'
                        self.sounds['chest_destroy'].play()
                        for i in range(100):
                            angle = random.random() * math.pi * 2
                            speed = random.random() * 2.5
                            self.sparks.append([[place_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2, place_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2], [math.cos(angle) * speed, math.sin(angle) * speed], random.random() * 3 + 3, 0.09, (12, 8, 2), False, 0.1])
                    continue
                if random.randint(1, 4) == 1:
                    side = random.choice([1, -1])
                    self.sparks.append([[tile[0] + self.TILE_SIZE * (side == 1), tile[1]], [random.random() * 0.1 - 0.05, random.random() * 0.5], random.random() * 5 + 3, 0.15, (4, 2, 12), False, 0])
                self.display.blit(self.tile_img, (tile[0], tile[1] + self.height))
                if tile[2] == 'chest':
                    self.display.blit(self.ghost_chest_img, (tile[0], tile[1] + self.height - self.TILE_SIZE))

            tile_rects = []
            for tile in self.tiles:
                self.display.blit(self.tile_img, (self.TILE_SIZE * tile[0], self.TILE_SIZE * tile[1] + int(self.height)))
                if self.tiles[tile] == 'placed_tile':
                    self.display.blit(self.placed_tile_img, (self.TILE_SIZE * tile[0], self.TILE_SIZE * tile[1] + int(self.height)))
                if self.tiles[tile] == 'chest':
                    self.display.blit(self.chest_img, (self.TILE_SIZE * tile[0], self.TILE_SIZE * (tile[1] - 1) + int(self.height)))
                    chest_r = pygame.Rect(self.TILE_SIZE * tile[0] + 2, self.TILE_SIZE * (tile[1] - 1) + 6, self.TILE_SIZE - 4, self.TILE_SIZE - 6)
                    if random.randint(1, 20) == 1:
                        self.sparks.append([[tile[0] * self.TILE_SIZE + 2 + 12 * random.random(), (tile[1] - 1) * self.TILE_SIZE + 4 + 8 * random.random()], [0, random.random() * 0.25 - 0.5], random.random() * 4 + 2, 0.023, (12, 8, 2), True, 0.002])
                    if chest_r.colliderect(self.player.rect):
                        self.sounds['chest_open'].play()
                        for i in range(50):
                            self.sparks.append([[tile[0] * self.TILE_SIZE + 8, (tile[1] - 1) * self.TILE_SIZE + 8], [random.random() * 2 - 1, random.random() - 2], random.random() * 3 + 3, 0.01, (12, 8, 2), True, 0.05])
                        self.tiles[tile] = 'opened_chest'
                        self.player.jumps += 1
                        self.player.attempt_jump()
                        self.player.velocity[1] = -3.5
                        if random.randint(1, 5) < 3:
                            self.items.append(Item(self, self.animation_manager, (tile[0] * self.TILE_SIZE + 5, (tile[1] - 1) * self.TILE_SIZE + 5), (6, 6), random.choice(['warp', 'cube', 'jump']), velocity=[random.random() * 5 - 2.5, random.random() * 2 - 5]))
                        else:
                            for i in range(random.randint(2, 6)):
                                self.items.append(Item(self, self.animation_manager, (tile[0] * self.TILE_SIZE + 5, (tile[1] - 1) * self.TILE_SIZE + 5), (6, 6), 'coin', velocity=[random.random() * 5 - 2.5, random.random() * 2 - 7]))
                elif self.tiles[tile] == 'opened_chest':
                    self.display.blit(self.opened_chest_img, (self.TILE_SIZE * tile[0], self.TILE_SIZE * (tile[1] - 1) + int(self.height)))

            base_row = max(self.tiles, key=lambda x: x[1])[1] - 1
            filled = True
            for i in range(self.WINDOW_TILE_SIZE[0] - 2):
                if (i + 1, base_row) not in self.tiles:
                    filled = False
            if filled:
                self.target_height = math.floor(self.height / self.TILE_SIZE) * self.TILE_SIZE + self.TILE_SIZE

            if self.height != self.target_height:
                self.height += (self.target_height - self.height) / 10
                if abs(self.target_height - self.height) < 0.2:
                    self.height = self.target_height
                    for i in range(self.WINDOW_TILE_SIZE[0] - 2):
                        if (i + 1, base_row) in self.tiles:
                            del self.tiles[(i + 1, base_row + 1)]

            for i in range(self.WINDOW_TILE_SIZE[1] + 2):
                pos_y = (-self.height // 16) - 1 + i
                self.display.blit(self.edge_tile_img, (0, self.TILE_SIZE * pos_y + int(self.height)))
                self.display.blit(self.edge_tile_img, (self.TILE_SIZE * (self.WINDOW_TILE_SIZE[0] - 1), self.TILE_SIZE * pos_y + int(self.height)))

            tile_rects.append(pygame.Rect(0, self.player.pos[1] - 300, self.TILE_SIZE, 600))
            tile_rects.append(pygame.Rect(self.TILE_SIZE * (self.WINDOW_TILE_SIZE[0] - 1), self.player.pos[1] - 300, self.TILE_SIZE, 600))

            for i, item in sorted(enumerate(self.items), reverse=True):
                lookup_pos = (int(item.center[0] // self.TILE_SIZE), int(item.center[1] // self.TILE_SIZE))
                if lookup_pos in self.tiles:
                    self.items.pop(i)
                    continue
                item.update(tile_rects + self.lookup_nearby(self.tiles, item.center))
                if item.time > 30:
                    if item.rect.colliderect(self.player.rect):
                        if item.type == 'coin':
                            self.sounds['coin'].play()
                            self.coins += 1
                            for j in range(25):
                                angle = random.random() * math.pi * 2
                                speed = random.random() * 0.4
                                physics_on = random.choice([False, False, False, False, True])
                                self.sparks.append([item.center.copy(), [math.cos(angle) * speed, math.sin(angle) * speed], random.random() * 3 + 3, 0.02, (12, 8, 2), physics_on, 0.1 * physics_on])
                        else:
                            self.sounds['collect_item'].play()
                            for j in range(50):
                                self.sparks.append([item.center.copy(), [random.random() * 0.3 - 0.15, random.random() * 6 - 3], random.random() * 4 + 3, 0.01, (12, 8, 2), False, 0])
                            self.current_item = item.type
                        self.items.pop(i)
                        continue
                r1 = int(9 + math.sin(self.master_clock / 30) * 3)
                r2 = int(5 + math.sin(self.master_clock / 40) * 2)
                self.display.blit(self.glow_img(r1, (12, 8, 2)), (item.center[0] - r1 - 1, item.center[1] + self.height - r1 - 2), special_flags=BLEND_RGBA_ADD)
                self.display.blit(self.glow_img(r2, (24, 16, 3)), (item.center[0] - r2 - 1, item.center[1] + self.height - r2 - 2), special_flags=BLEND_RGBA_ADD)
                item.render(self.display, (0, -self.height))

            if not self.dead:
                self.player.update(tile_drop_rects + tile_rects + self.lookup_nearby(self.tiles, self.player.center))
            else:
                self.player.opacity = 80
                self.player.update([])
                self.player.rotation -= 16
            self.player.render(self.display, (0, -int(self.height)))

            for i, spark in sorted(enumerate(self.sparks), reverse=True):
                # pos, vel, size, decay, color, physics, gravity, dead

                if len(spark) < 8:
                    spark.append(False)

                if not spark[-1]:
                    spark[1][1] = min(spark[1][1] + spark[-2], 3)

                spark[0][0] += spark[1][0]
                if spark[5]:
                    if ((int(spark[0][0] // self.TILE_SIZE), int(spark[0][1] // self.TILE_SIZE)) in self.tiles) or (spark[0][0] < self.TILE_SIZE) or (spark[0][0] > self.DISPLAY_SIZE[0] - self.TILE_SIZE):
                        spark[0][0] -= spark[1][0]
                        spark[1][0] *= -0.7
                spark[0][1] += spark[1][1]
                if spark[5]:
                    if (int(spark[0][0] // self.TILE_SIZE), int(spark[0][1] // self.TILE_SIZE)) in self.tiles:
                        spark[0][1] -= spark[1][1]
                        spark[1][1] *= -0.7
                        if abs(spark[1][1]) < 0.1:
                            spark[1][1] = 0
                            spark[-1] = True
                #if spark[-2]:
                #    spark[1][0] = normalize(spark[1][0], 0.03)
                spark[2] -= spark[3]
                if spark[2] <= 1:
                    self.sparks.pop(i)
                else:
                    self.display.blit(self.glow_img(int(spark[2] * 1.5 + 2), (int(spark[4][0] / 2), int(spark[4][1] / 2), int(spark[4][2] / 2))), (spark[0][0] - spark[2] * 2, spark[0][1] + self.height - spark[2] * 2), special_flags=BLEND_RGBA_ADD)
                    self.display.blit(self.glow_img(int(spark[2]), spark[4]), (spark[0][0] - spark[2], spark[0][1] + self.height - spark[2]), special_flags=BLEND_RGBA_ADD)
                    self.display.set_at((int(spark[0][0]), int(spark[0][1] + self.height)), (255, 255, 255))

            self.display.blit(self.border_img_light, (0, -math.sin(self.master_clock / 30) * 4 - 7))
            self.display.blit(self.border_img, (0, -math.sin(self.master_clock / 40) * 7 - 14))
            self.display.blit(pygame.transform.flip(self.border_img_light, False, True), (0, self.DISPLAY_SIZE[1] + math.sin(self.master_clock / 40) * 3 + 9 - self.border_img.get_height()))
            self.display.blit(pygame.transform.flip(self.border_img, False, True), (0, self.DISPLAY_SIZE[1] + math.sin(self.master_clock / 30) * 3 + 16 - self.border_img.get_height()))

            # UI
            if not self.dead:
                self.display.blit(self.coin_icon, (4, 4))
                self.black_font.render(str(self.coins), self.display, (13, 6))
                self.white_font.render(str(self.coins), self.display, (12, 5))

                self.display.blit(self.item_slot_img, (self.DISPLAY_SIZE[0] - 20, 4))
                if self.current_item:
                    if (self.master_clock % 50 < 12) or (abs(self.master_clock % 50 - 20) < 3):
                        self.display.blit(self.item_slot_flash_img, (self.DISPLAY_SIZE[0] - 20, 4))
                    self.display.blit(self.item_icons[self.current_item], (self.DISPLAY_SIZE[0] - 15, 9))
                    if not self.item_used:
                        if (self.master_clock % 100 < 80) or (abs(self.master_clock % 100 - 90) < 3):
                            self.black_font.render('press E/X to use', self.display, (self.DISPLAY_SIZE[0] - self.white_font.width('press E/X to use') - 23, 10))
                            self.white_font.render('press E/X to use', self.display, (self.DISPLAY_SIZE[0] - self.white_font.width('press E/X to use') - 24, 9))
            else:
                self.black_font.render('game over', self.display, (self.DISPLAY_SIZE[0] // 2 - self.white_font.width('game over') // 2 + 1, 51))
                self.white_font.render('game over', self.display, (self.DISPLAY_SIZE[0] // 2 - self.white_font.width('game over') // 2, 50))
                coin_count_width = self.white_font.width(str(self.end_coin_count))
                self.display.blit(self.coin_icon, (self.DISPLAY_SIZE[0] // 2 - (coin_count_width + 4 + self.coin_icon.get_width()) // 2, 63))
                self.black_font.render(str(self.end_coin_count), self.display, ((self.DISPLAY_SIZE[0] + 5 + self.coin_icon.get_width()) // 2 - coin_count_width // 2, 65))
                self.white_font.render(str(self.end_coin_count), self.display, ((self.DISPLAY_SIZE[0] + 4 + self.coin_icon.get_width()) // 2 - coin_count_width // 2, 64))
                if self.master_clock % 3 == 0:
                    if self.end_coin_count != self.coins:
                        self.sounds['coin_end'].play()
                    self.end_coin_count = min(self.end_coin_count + 1, self.coins)
                if (self.master_clock % 100 < 80) or (abs(self.master_clock % 100 - 90) < 3):
                    self.black_font.render('press R to restart', self.display, (self.DISPLAY_SIZE[0] // 2 - self.white_font.width('press R to restart') // 2 + 1, 79))
                    self.white_font.render('press R to restart', self.display, (self.DISPLAY_SIZE[0] // 2 - self.white_font.width('press R to restart') // 2, 78))

            if self.handling_music and self.handling_music.done():
                self.handling_music = None
            
            if self.submitting_music and self.submitting_music.done():
                self.submitting_music = None
                        
            if not self.handling_music and not self.midi_player.music_out_priority_queue.empty():
                self.handling_music = executor.submit(self.midi_player.handle_music_events, self.current_item, self.dead)
                
            if not self.submitting_music and (self.midi_player.g_timestamp - pygame.midi.time() < 500 or self.midi_player.music_out_priority_queue.qsize() < 50):
                self.submitting_music = executor.submit(self.midi_player.gen_new_music, self.coins)
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    executor.shutdown(wait=False)
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key in [K_RIGHT, K_d]:
                        self.player.right = True
                    if event.key in [K_LEFT, K_a]:
                        self.player.left = True
                    if event.key in [K_UP, K_w, K_SPACE]:
                        if not self.dead:
                            if self.player.jumps:
                                self.sounds['jump'].play()
                            self.player.attempt_jump()
                    if event.key in [K_e, K_x]:
                        if not self.dead:
                            if self.current_item:
                                self.item_used = True
                            if self.current_item == 'warp':
                                self.sounds['warp'].play()
                                max_point = min(enumerate(self.stack_heights), key=lambda x: x[1])
                                for i in range(60):
                                    angle = random.random() * math.pi * 2
                                    speed = random.random() * 3
                                    physics_on = random.choice([False, True])
                                    self.sparks.append([self.player.center.copy(), [math.cos(angle) * speed, math.sin(angle) * speed], random.random() * 3 + 3, 0.02, (12, 8, 2), physics_on, 0.1 * physics_on])
                                self.player.pos[0] = (max_point[0] + 1) * self.TILE_SIZE + 4
                                self.player.pos[1] = (max_point[1] - 1) * self.TILE_SIZE
                                for i in range(60):
                                    angle = random.random() * math.pi * 2
                                    speed = random.random() * 1.75
                                    physics_on = random.choice([False, True, True])
                                    self.sparks.append([self.player.center.copy(), [math.cos(angle) * speed, math.sin(angle) * speed], random.random() * 3 + 3, 0.02, (12, 8, 2), physics_on, 0.1 * physics_on])
                            if self.current_item == 'jump':
                                self.sounds['super_jump'].play()
                                self.player.jumps += 1
                                self.player.attempt_jump()
                                self.player.velocity[1] = -8
                                for i in range(60):
                                    angle = random.random() * math.pi / 2 + math.pi / 4
                                    if random.randint(1, 5) == 1:
                                        angle = -math.pi / 2
                                    speed = random.random() * 3
                                    physics_on = random.choice([False, True])
                                    self.sparks.append([self.player.center.copy(), [math.cos(angle) * speed, math.sin(angle) * speed], random.random() * 3 + 3, 0.02, (12, 8, 2), physics_on, 0.1 * physics_on])
                            if self.current_item == 'cube':
                                self.sounds['block_land'].play()
                                place_pos = (int(self.player.center[0] // self.TILE_SIZE), int(self.player.pos[1] // self.TILE_SIZE) + 1)
                                self.stack_heights[place_pos[0] - 1] = place_pos[1]
                                for i in range(place_pos[1], base_row + 2):
                                    self.tiles[(place_pos[0], i)] = 'placed_tile'
                                    for j in range(8):
                                        self.sparks.append([[place_pos[0] * self.TILE_SIZE + self.TILE_SIZE, i * self.TILE_SIZE + j * 2], [random.random() * 0.5, random.random() * 0.5 - 0.25], random.random() * 4 + 4, 0.02, (12, 8, 2), False, 0])
                                        self.sparks.append([[place_pos[0] * self.TILE_SIZE, i * self.TILE_SIZE + j * 2], [-random.random() * 0.5, random.random() * 0.5 - 0.25], random.random() * 4 + 4, 0.02, (12, 8, 2), False, 0])
                            self.current_item = None
                    if event.key == K_r:
                        if self.dead:
                            self.player = Player(self, self.animation_manager, (self.DISPLAY_SIZE[0] // 2 - 5, -20), (8, 16), 'player')
                            self.dead = False

                            self.tiles = {}

                            self.tile_drops = []

                            self.sparks = []

                            self.game_timer = 0
                            self.height = 0
                            self.target_height = 0
                            self.coins = 0
                            self.end_coin_count = 0
                            self.current_item = None
                            self.master_clock = 0

                            with self.midi_player.music_out_priority_queue.mutex:
                                self.midi_player.music_out_priority_queue.queue.clear()
                                self.midi_player.music_out_priority_queue.all_tasks_done.notify_all()
                                self.midi_player.music_out_priority_queue.unfinished_tasks = 0
                            
                            for i in range(self.WINDOW_TILE_SIZE[0] - 2):
                                self.tiles[(i + 1, self.WINDOW_TILE_SIZE[1] - 1)] = 'tile'

                            self.tiles[(1, self.WINDOW_TILE_SIZE[1] - 2)] = 'tile'
                            self.tiles[(self.WINDOW_TILE_SIZE[0] - 2, self.WINDOW_TILE_SIZE[1] - 2)] = 'tile'

                            self.stack_heights = [self.WINDOW_TILE_SIZE[1] - 1 for i in range(self.WINDOW_TILE_SIZE[0] - 2)]
                            self.stack_heights[0] -= 1
                            self.stack_heights[-1] -= 1

                            self.items = []

                if event.type == KEYUP:
                    if event.key in [K_RIGHT, K_d]:
                        self.player.right = False
                    if event.key in [K_LEFT, K_a]:
                        self.player.left = False

            self.screen.blit(pygame.transform.scale(self.display, self.screen.get_size()), (0, 0))
            pygame.display.update()
            self.clock.tick(60)
    
if __name__ == '__main__':
    game = Game()
    game.run()
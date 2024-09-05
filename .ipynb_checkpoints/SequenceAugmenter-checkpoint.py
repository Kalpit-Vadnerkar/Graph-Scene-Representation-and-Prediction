import numpy as np
from copy import deepcopy

class SequenceAugmenter:
    def __init__(self, sequence):
        self.sequence = deepcopy(sequence)

    def _translate_and_scale_positions(self):
        # Collect all positions
        all_positions = []
        for node in self.sequence['graph'].nodes(data=True):
            all_positions.append([node[1]['x'], node[1]['y']])
        for timestep in self.sequence['past'] + self.sequence['future']:
            all_positions.append(timestep['position'])
            for obj in timestep['objects']:
                all_positions.append(obj['position'])
        
        all_positions = np.array(all_positions)

        # Find min and max x and y values
        min_pos = np.min(all_positions, axis=0)
        max_pos = np.max(all_positions, axis=0)

        # Calculate range for x and y
        pos_range = max_pos - min_pos

        # Define translation and scaling function
        def translate_and_scale(pos):
            return (np.array(pos) - min_pos) / pos_range

        # Apply translation and scaling to graph nodes
        for node in self.sequence['graph'].nodes(data=True):
            scaled_pos = translate_and_scale([node[1]['x'], node[1]['y']])
            node[1]['x'], node[1]['y'] = scaled_pos

        # Apply translation and scaling to past and future positions
        for timestep in self.sequence['past'] + self.sequence['future']:
            timestep['position'] = translate_and_scale(timestep['position']).tolist()
            for obj in timestep['objects']:
                obj['position'] = translate_and_scale(obj['position']).tolist()

    def rotate(self, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        rot_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])

        # Rotate graph nodes
        for node in self.sequence['graph'].nodes(data=True):
            pos = np.array([node[1]['x'], node[1]['y']])
            rotated_pos = np.dot(rot_matrix, pos)
            node[1]['x'], node[1]['y'] = rotated_pos

        # Rotate past and future positions
        for timestep in self.sequence['past'] + self.sequence['future']:
            pos = np.array(timestep['position'])
            rotated_pos = np.dot(rot_matrix, pos)
            timestep['position'] = rotated_pos.tolist()

            # Rotate object positions
            for obj in timestep['objects']:
                obj_pos = np.array(obj['position'])
                rotated_obj_pos = np.dot(rot_matrix, obj_pos)
                obj['position'] = rotated_obj_pos.tolist()

        # Translate and scale positions to [0, 1] after rotation
        self._translate_and_scale_positions()

        return self.sequence

    def mirror(self, axis='x'):
        # Mirror graph nodes
        for node in self.sequence['graph'].nodes(data=True):
            if axis == 'x':
                node[1]['y'] = 1 - node[1]['y']
            else:
                node[1]['x'] = 1 - node[1]['x']

        # Mirror past and future positions
        for timestep in self.sequence['past'] + self.sequence['future']:
            if axis == 'x':
                timestep['position'][1] = 1 - timestep['position'][1]
            else:
                timestep['position'][0] = 1 - timestep['position'][0]

            # Mirror object positions
            for obj in timestep['objects']:
                if axis == 'x':
                    obj['position'][1] = 1 - obj['position'][1]
                else:
                    obj['position'][0] = 1 - obj['position'][0]

        return self.sequence

    def augment(self, rotations=None, mirrors=None):
        augmented_sequences = []

        # Perform rotations
        if rotations:
            for angle in rotations:
                rotated_sequence = self.rotate(angle)
                augmented_sequences.append(deepcopy(rotated_sequence))

        # Perform mirroring
        if mirrors:
            for axis in mirrors:
                mirrored_sequence = self.mirror(axis)
                augmented_sequences.append(deepcopy(mirrored_sequence))

        return augmented_sequences
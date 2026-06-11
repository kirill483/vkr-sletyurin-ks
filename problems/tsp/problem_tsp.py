from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP


class TSP(object):
    NAME = "tsp"

    @staticmethod
    def get_costs(dataset, pi):
        """
        dataset:
            {
                'templates': [B, N, K, 5],
                'depot':     [B, 2]
            }

        templates[..., 0:2] = entry point
        templates[..., 2:4] = exit point
        templates[..., 4]   = coverage cost

        pi:
            [B, N]
            values are in 1..N*K
        """
        templates = dataset["templates"]
        depot = dataset["depot"]

        batch_size, n_fields, n_templates, _ = templates.size()

        selected = pi - 1
        field_ids = selected // n_templates
        template_ids = selected % n_templates

        assert (
            torch.arange(
                n_fields,
                device=pi.device,
                dtype=field_ids.dtype,
            ).view(1, -1).expand_as(field_ids)
            == field_ids.sort(1)[0]
        ).all(), "Invalid tour: each field must be selected exactly once"

        batch_idx = torch.arange(batch_size, device=pi.device)[:, None].expand(batch_size, n_fields)

        chosen_templates = templates[
            batch_idx,
            field_ids,
            template_ids,
        ]  # [B, N, 5]

        chosen_in = chosen_templates[..., 0:2]
        chosen_out = chosen_templates[..., 2:4]
        chosen_coverage = chosen_templates[..., 4]

        depot_to_first = (chosen_in[:, 0] - depot).norm(p=2, dim=1)
        between_templates = (chosen_in[:, 1:] - chosen_out[:, :-1]).norm(p=2, dim=2).sum(1)
        last_to_depot = (chosen_out[:, -1] - depot).norm(p=2, dim=1)

        transition_cost = depot_to_first + between_templates + last_to_depot
        coverage_cost = chosen_coverage.sum(1)

        return transition_cost + coverage_cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(input):
        return StateTSP.initialize(input)


class TSPDataset(Dataset):

    def __init__(self, filename, num_samples=1000000, offset=0, **kwargs):
        super(TSPDataset, self).__init__()

        assert os.path.splitext(filename)[1] == ".pkl"

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.data = [
            {
                "depot": torch.FloatTensor(depot),
                "templates": torch.FloatTensor(templates),
            }
            for depot, templates in data[offset: offset + num_samples]
        ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
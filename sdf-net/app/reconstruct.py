import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from skimage import measure
import trimesh
from lib.models.OctreeSDF import OctreeSDF
from lib.options import parse_options

if __name__ == '__main__':
    
    parser = parse_options(return_parser=True)
    args = parser.parse_args()
    lod = args.lod
    net = OctreeSDF(args)

    if args.pretrained is not None:
        name = args.pretrained.split('/')[-1].split('.')[0]
    else:
        assert False and "No network weights specified!"

    net.load_state_dict(torch.load(args.pretrained))

    device = torch.device('cuda')
    net.to(device)
    net.eval()

    # Define the grid over which to evaluate the SDF.
    # Adjust the bounds and grid resolution as needed.
    grid_size = args.render_res[0]
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    zmin, zmax = -1, 1

    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)
    z = np.linspace(zmin, zmax, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1)  # shape (grid_size, grid_size, grid_size, 3)

    # Flatten grid to shape (num_points, 3) and convert to a torch tensor.
    points_flat = points.reshape(-1, 3)
    points_tensor = torch.tensor(points_flat, device=device, dtype=torch.float32)

    # Evaluate the SDF using your NGLOD model.
    print('lod: ', lod)
    with torch.no_grad():
        sdf_values = net(points_tensor, lod=lod)  # Assume output shape (num_points,)
    sdf_values = sdf_values.cpu().numpy().reshape((grid_size, grid_size, grid_size))

    # Use marching cubes to extract the 0-levelset surface.
    # The spacing parameter converts voxel indices into world coordinates.
    spacing = ((x[1]-x[0]), (y[1]-y[0]), (z[1]-z[0]))
    vertices, faces, normals, _ = measure.marching_cubes(sdf_values, level=0, spacing=spacing)

    # Adjust vertices by adding the grid origin (xmin, ymin, zmin)
    vertices += np.array([xmin, ymin, zmin])

    # Create a mesh using trimesh and export it (e.g., as OBJ).
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    
    output_dir = "_results/meshes/{}/".format(args.exp_name)
    output_file = output_dir + 'mesh.obj'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mesh.export(output_file)
    print("Mesh exported to {}".format(output_file))
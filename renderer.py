import numpy as np
import torch
import imageio
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer,
    FoVPerspectiveCameras, look_at_view_transform,
    MeshRenderer, HardPhongShader, PointLights, AmbientLights,
    TexturesVertex
)
import cv2

def get_simple_rasterizer(img_size=512, faces_per_pixel=1, cull_backfaces=True, blur_radius=0):
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=blur_radius, 
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=cull_backfaces ,
        bin_size = None,
    )
    return MeshRasterizer(raster_settings=raster_settings)

def get_camera(dist, elev, azim, T, znear=0.1, zfar=1000, device='cpu'):
    R, T_ = look_at_view_transform(dist, elev, azim)
    T = T_ + T
    camera = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar)

    return camera

def get_simple_renderer(img_size=512, device='cuda', type='point', location=None):
    location = [4., 0., 4.0] if location is None else location
    if type == 'point':
        lights = PointLights(device=device, location=[location])
    else:
        lights = AmbientLights(device=device)
    rasterizer = get_simple_rasterizer(img_size=img_size)
    shader = HardPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )

    return renderer, rasterizer

class Renderer(object):
    def __init__(self,
                 img_size=(512, 512),
                 device='cpu'):
        super().__init__()
        self.device = device
        self.img_size = img_size

        self.renderer, self.rasterizer = get_simple_renderer(img_size=self.img_size, device=self.device)
        self.renderer2x, _ = get_simple_renderer(img_size=[img_size[0] * 2, img_size[0] * 2], device=self.device)

    @torch.no_grad()
    def render(self, vertices, faces, camera, col=[0.35, 0.35, 0.35], aa=False):
        meshes = []
        for v, f in zip(vertices, faces):
            cols = torch.ones_like(v[0], device=v.device) * torch.tensor(col, device=v.device)
            cols = cols.to(v.dtype)
            textures = TexturesVertex(verts_features=cols)
            mesh = Meshes(verts=v[0], faces=f, textures=textures)
            meshes.append(mesh)
        meshes = join_meshes_as_scene(meshes)
        if aa:
            img = self.renderer2x(meshes, cameras=camera)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
            alpha = 1 - torch.nn.functional.max_pool2d(1 - img[:, -1:,...], kernel_size=2, stride=2)
            img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2)
            img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
            alpha = alpha.permute(0, 2, 3, 1)
        else:
            img = self.renderer(meshes, cameras=camera)
            alpha = img[..., -1:]

        return img[0], alpha[0]

    def draw_points_on_img(self, img, points, color=[0, 1, 0], radius=3):
        if len(points) == 0:
            return img
        
        assert img.ndim == 3, "H X W X C"
        assert points.ndim == 2, "N X 2"
        if img.shape[-1] == 4:
            assert len(color) == 4, "Color should be RGBA if img is RGBA"
    
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        img = np.copy(img)
        points = points.astype(np.int32)
        
        for point in points:
            point = point.astype(np.int32)
            cv2.circle(img, tuple(point), radius, color, -1)
        
        return img
    
    def draw_lines_on_img(self, img, lines, color=[0, 1, 0], thickness=1):
        assert img.ndim == 3, "H X W X C"
        assert lines.ndim == 3, "N X 2 X 2"
        if img.shape[-1] == 4:
            assert len(color) == 4, "Color should be RGBA if img is RGBA"
    
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(lines, torch.Tensor):
            lines = lines.cpu().numpy()
        
        img = np.copy(img)
        lines = lines.astype(np.int32)
        
        for line in lines:
            line = line.astype(np.int32)
            cv2.line(img, tuple(line[0]), tuple(line[1]), color, thickness)
        
        return img
    
    def save_video(self, tensor, path, fps, loop=True, bg_col=[1,1,1]):
        if tensor.max() <= 1:
            tensor = tensor * 255
        alpha = tensor[..., -1:] / 255
        img = tensor[..., :-1]
        tensor = img * alpha + (1 - alpha) * torch.tensor(bg_col, device=self.device)[None,None,None] * 255.

        video = tensor.cpu().numpy().astype(np.uint8)
        video_writer = imageio.get_writer(path, fps=fps, quality=10)
        for j in range(len(video)):
            video_writer.append_data(video[j])
        if loop:
            for j in range(len(video)-1,-1,-1):
                video_writer.append_data(video[j])
        video_writer.close()

    def save_image(self, tensor, path):
        assert tensor.ndim == 3, "H X W X C"
        if tensor.max() <= 1:
            tensor = tensor * 255
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy().astype(np.uint8)
        image = tensor.astype(np.uint8)
        imageio.imwrite(path, image)

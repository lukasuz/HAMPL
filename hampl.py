from renderer import Renderer, get_camera
from pytorch3d.io import save_obj
import torch
from utils import *
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--character', default='火', type=str)
    parser.add_argument('-s', '--size', help='image size', nargs='+', type=int, default=[128, 128])
    parser.add_argument('-d', '--device', help='device', default='cpu', type=str)
    parser.add_argument('--smpl_path', help='SMPL path', default='SMPL_MALE.pkl', type=str)
    parser.add_argument('--graphics_path', help='Graphics path', default='graphics.txt', type=str)
    parser.add_argument('--no_render', help='Render', action='store_true')
    parser.add_argument('--save_mesh', help='Save mesh', action='store_true')

    # Optimization params
    parser.add_argument('--alpha', help='Stroke alpha, how many points are sample on each stroke for the chamfer loss', default=0.5, type=float)
    parser.add_argument('--iters', help='Optimization iters per stroke', default=1000, type=int)
    parser.add_argument('--tensorboard_path', help='Tensorboard path', default=None, type=str)
    parser.add_argument('--verbose', help='Verbose', action='store_true')
    parser.add_argument('--seed', help='Random seed', default=0, type=int)

    # Video params
    parser.add_argument('--fps', help='Video fps', default=15, type=int)
    parser.add_argument('--no_loop', help='Loop video', action='store_true')
    parser.add_argument('--smpl_col', help='SMPL color', nargs='+', type=int, default=[80, 80, 80])
    parser.add_argument('--bg_col', help='Background color', nargs='+', type=int, default=[236, 240, 241])
    parser.add_argument('--num_rotation_frames', help='Number of rotation frames', default=10, type=int)
    parser.add_argument('--num_blend_frames', help='Number of blend frames', default=10, type=int)
    parser.add_argument('--num_pause_frames', help='Number of pause frames', default=10, type=int)
    args = parser.parse_args()

    character = args.character
    img_size = args.size
    device = args.device
    smpl_path = args.smpl_path
    graphics_path = args.graphics_path

    alpha = args.alpha
    iters = args.iters
    tensorboard_path = args.tensorboard_path
    verbose = args.verbose
    seed = args.seed

    fps = args.fps
    loop = not args.no_loop
    smpl_col = args.smpl_col
    bg_col = args.bg_col
    num_rotation_frames = args.num_rotation_frames
    num_blend_frames = args.num_blend_frames
    num_pause_frames = args.num_pause_frames

    os.makedirs('outputs', exist_ok=True)

    bg_col = np.array(bg_col) / 255
    smpl_col = np.array(smpl_col) / 255

    side_cam_kwargs = {
        'dist': 2.,
        'elev': 0,
        'azim': 90,
        'T': torch.tensor([0., 0., 0.])
    }
    camera = get_camera(*side_cam_kwargs.values()).to(device)
    camera.image_size = img_size
    torch.manual_seed(seed)
    renderer = Renderer(img_size=img_size, device=device)
    smpl = SMPL_Wrapper(smpl_path, device=device)
    stroke = Stroke_Wrapper(graphics_path)
    stroke_img = stroke.get_img(character, img_size=img_size)

    all_vertices = []
    all_faces = []
    T = []
    for out in stroke(character, img_size=img_size, alpha=alpha):
        stroke_points, extra_stroke_points, stroke_lines, c_str = out
        print('Optimizing stroke', c_str)
        all_stroke_points = torch.cat([stroke_points, extra_stroke_points], dim=0).to(device)
        pose_opt, rot_opt, transl_opt = smpl.optimize_pose(camera, all_stroke_points, iters=iters+1, log_iter=100, renderer=renderer, tensorboard_path=tensorboard_path, verbose=verbose, alpha=alpha)
        T.append(transl_opt)
        with torch.no_grad():
            vertices, faces, points, extra_points, lines, _ = smpl(camera, pose_opt, rot_opt, transl_opt)
            if args.save_mesh:
                _c_str = c_str.replace('/', '-')
                save_obj(f'outputs/{character}_{img_size[0]}_{_c_str}.obj', vertices[0,0], faces[0])
            all_vertices.append(vertices)
            all_faces.append(faces)

        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    if not args.no_render: 
        T = torch.concat(T, dim=0)
        with torch.no_grad():
            final_img, alpha = renderer.render(all_vertices, all_faces, camera, col=smpl_col, aa=True)
            final_img_mask = alpha > 0
            renderer.save_image(final_img, f'outputs/{character}_{img_size[0]}.png')

            stroke_img = torch.tensor(stroke_img).to(device).float() / 255
            stroke_alpha = stroke_img[:,:,3]
            stroke_img = cv2.inpaint(
                final_img[:,:,:3].mul(255).cpu().numpy().astype(np.uint8), 
                (1 - final_img_mask.to(torch.float32)).cpu().numpy().astype(np.uint8), 
                5, cv2.INPAINT_TELEA)
            stroke_img = torch.tensor(stroke_img).to(device).float() / 255
            stroke_img = torch.concat([stroke_img, stroke_alpha[:,:,None]], dim=-1)
            
            blend_images = []
            for i in np.linspace(0, 1, num_blend_frames):
                img = (1 - i) * final_img + i * stroke_img
                blend_images.append(img)

            # renderer.save_video(
            #     torch.stack(blend_images, dim=0), 
            #     f'outputs/{character}_{img_size[0]}_blend.mp4', 
            #     fps=fps,
            #     bg_col=bg_col,
            #     loop=False)

            imgs = []
            azims = np.linspace(0, 90, num_rotation_frames).astype(int)
            dists = np.linspace(5, 2, num_rotation_frames)
            x_range = (T[:,0].max() - T[:,0].min()).cpu().numpy()
            y_range = (T[:,1].max() - T[:,1].min()).cpu().numpy()
            Xs = np.linspace(x_range / 2, 0, num_rotation_frames)
            Ys = np.linspace(y_range / 2, 0, num_rotation_frames)
            for azim, dist, x, y in zip(azims[:-1], dists[:-1], Xs[:-1], Ys):
                print("Rendering angle", azim)
                cam_kwargs = {
                    'dist': dist,
                    'elev': 0,
                    'azim': azim,
                    'T': torch.tensor([-x, 0, 0.])
                }
                camera = get_camera(*cam_kwargs.values())
                camera.image_size = img_size
                camera = camera.to(device)
                img, _ = renderer.render(all_vertices, all_faces, camera, col=smpl_col, aa=True)
                imgs.append(img)

            imgs.append(final_img)
            imgs = [imgs[0]] * num_pause_frames + imgs + [imgs[-1]] * num_pause_frames
            imgs = imgs + blend_images
            imgs = imgs + [imgs[-1]] * num_pause_frames
            img = torch.stack(imgs, dim=0)
            renderer.save_video(
                torch.stack(imgs, dim=0), 
                f'outputs/{character}_{img_size[0]}.mp4', 
                fps=fps, 
                loop=loop,
                bg_col=bg_col)
import os
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def charger_modele_star(npz_path):
    """
    Charge un fichier STAR .npz, retourne le template vertices et la matrice de régression des joints.
    """
    data = np.load(npz_path)
    v_template = data['v_template']  # (N_vertices, 3)
    J_regressor = data['J_regressor']  # (N_joints, N_vertices)
    return v_template, J_regressor

def calculer_joints(v_template, J_regressor):
    """
    Calcule les positions des joints Jtr par multiplication matricielle.
    v_template: (N_vertices, 3)
    J_regressor: (N_joints, N_vertices)
    Retourne : Jtr (N_joints, 3)
    """
    Jtr = J_regressor.dot(v_template)
    return Jtr

def afficher_indices_joints(Jtr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Jtr[:, 0], Jtr[:, 1], Jtr[:, 2], c='gray', s=40)
    for idx, (x, y, z) in enumerate(Jtr):
        ax.text(x, y, z, str(idx), color='blue', fontsize=10)
    ax.set_title("Indices des joints STAR")
    plt.show()


def visualiser_joints_squelette(Jtr, titre="Squelette 3D des joints"):
    """
    Visualise les joints et leurs connexions en 3D avec matplotlib.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (0, 6),
        (5, 7), (6, 8),
        (2, 9), (9, 10), (10, 11), (11, 12)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracer les joints
    ax.scatter(Jtr[:, 0], Jtr[:, 1], Jtr[:, 2], c='gray', s=40)

    # Afficher les indices des joints
    for idx, (x, y, z) in enumerate(Jtr):
        ax.text(x, y, z, str(idx), color='blue', fontsize=8)

    # Tracer les connexions
    for i, j in edges:
        if i < len(Jtr) and j < len(Jtr):
            x = [Jtr[i, 0], Jtr[j, 0]]
            y = [Jtr[i, 1], Jtr[j, 1]]
            z = [Jtr[i, 2], Jtr[j, 2]]
            ax.plot(x, y, z, 'r-')

    ax.set_title(titre)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Calcul des limites avec margin
    margin = 0.1
    x_min, x_max = Jtr[:, 0].min() - margin, Jtr[:, 0].max() + margin
    y_min, y_max = Jtr[:, 1].min() - margin, Jtr[:, 1].max() + margin
    z_min, z_max = Jtr[:, 2].min() - margin, Jtr[:, 2].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Centrer la vue
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    ax.view_init(elev=20, azim=45)  # ajuster angle caméra

    plt.show()


def visualiser_joints(Jtr, titre="Visualisation des joints STAR"):

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])

    # Créer une sphère rouge pour chaque joint
    sphere_radius = 0.01  # tu peux l'augmenter si tu ne vois rien
    for joint in Jtr:
        sphere = trimesh.creation.icosphere(radius=sphere_radius)
        sphere.apply_translation(joint)
        mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
        scene.add(mesh)

    # Calcul du centre pour la caméra
    center = Jtr.mean(axis=0)
    distance = 0.5  # distance caméra

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, center[0]],
        [0.0, 1.0, 0.0, center[1]],
        [0.0, 0.0, 1.0, center[2] + distance],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    pyrender.Viewer(scene, use_raymond_lighting=True, window_title=titre)
    


def visualiser_joints_sur_mesh(v_template, Jtr, genre="neutral", titre="Joints sur mesh STAR"):
    """
    Visualise les joints superposés au mesh STAR avec pyrender.
    genre: "female", "male" ou "neutral" pour charger le bon fichier faces.
    Affiche aussi les liens ("bones") entre joints pour mieux voir la structure.
    """

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.2, 0.2, 0.2])

    # Chargement faces
    faces_path = os.path.join(os.path.dirname(__file__), '..', 'star_1_1', genre, 'smplh_faces.npy')
    if not os.path.exists(faces_path):
        print(f"❌ Fichier des faces manquant ({faces_path})")
        return
    faces = np.load(faces_path)
    mesh = trimesh.Trimesh(vertices=v_template, faces=faces, process=False)
    star_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(star_mesh)

    # Debug : afficher quelques positions joints
    print("Positions joints (extraits):", Jtr[:5])

    # Couleur jaune vif
    yellow_color = np.array([255, 255, 0, 255], dtype=np.uint8)
    sphere_radius = 0.04

    # Ajouter les joints comme sphères jaunes
    for joint in Jtr:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        sphere.apply_translation(joint)
        sphere.visual.vertex_colors = yellow_color
        joint_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
        scene.add(joint_mesh)

    # Liste des liens ("bones") entre joints STAR (indices)
    bones = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7),
        (0,8), (8,9), (9,10),
        (0,11), (11,12), (12,13),
        (0,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20), (20,21), (21,22), (22,23)
    ]

    # Ajouter des cylindres entre joints pour les bones
    for (i_start, i_end) in bones:
        if i_start >= len(Jtr) or i_end >= len(Jtr):
            continue
        start = Jtr[i_start]
        end = Jtr[i_end]
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        direction = vec / length

        # Création cylindre aligné avec le segment
        cyl_radius = sphere_radius / 3
        cylinder = trimesh.creation.cylinder(radius=cyl_radius, height=length, sections=16)

        # Alignement du cylindre sur le vecteur
        axis = np.array([0, 0, 1])
        # Rotation qui aligne axis sur direction
        rot_quat = R.align_vectors([direction], [axis])[0]
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = rot_quat.as_matrix()
        cylinder.apply_transform(rot_mat)

        # Positionner le cylindre entre les deux joints
        translation = start + vec / 2
        cylinder.apply_translation(translation)

        # Couleur jaune aussi
        cylinder.visual.vertex_colors = yellow_color
        bone_mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=True)
        scene.add(bone_mesh)

    # Afficher axes XYZ pour orientation
    axes = trimesh.creation.axis(axis_length=0.1)
    axes_mesh = pyrender.Mesh.from_trimesh(axes, smooth=False)
    scene.add(axes_mesh)

    # Caméra centrée et reculée
    center = v_template.mean(axis=0)
    distance = 2.0
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, center[0]],
        [0.0, 1.0, 0.0, center[1]],
        [0.0, 0.0, 1.0, center[2] + distance],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # Lumière directionnelle
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)

    # Affichage
    pyrender.Viewer(scene, use_raymond_lighting=True, window_title=titre)

def pipeline_star():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'star_1_1')

    fichiers = {
        'female': os.path.join(base_dir, 'female', 'female.npz'),
        'male': os.path.join(base_dir, 'male', 'male.npz'),
        'neutral': os.path.join(base_dir, 'neutral', 'neutral.npz')
    }

    for genre, path_npz in fichiers.items():
        print(f"\n📌 Traitement du modèle {genre.upper()}")
        if not os.path.exists(path_npz):
            print(f"❌ Fichier non trouvé : {path_npz}")
            continue

        v_template, J_regressor = charger_modele_star(path_npz)
        Jtr = calculer_joints(v_template, J_regressor)

        print(f"Nombre de joints : {Jtr.shape[0]}")
        print(f"Positions joints (extraits):\n{Jtr[:5]} ...")

        visualiser_joints(Jtr, titre=f"Joints STAR - {genre}")
        visualiser_joints_squelette(Jtr, titre=f"Squelette 3D - {genre}")
        afficher_indices_joints(Jtr)
        visualiser_joints_sur_mesh(v_template, Jtr, titre=f"Joints + Mesh - {genre}")



if __name__ == "__main__":
    pipeline_star()

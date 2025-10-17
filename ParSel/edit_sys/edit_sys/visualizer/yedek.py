    # def _ray_pick(self, x, y):
    #     """Return (hit_bool, [x, y, z]) that works on all Open3D ≤ 0.18."""
    #     s   = self._scene.scene          # Open3DScene
    #     cam = s.camera

    #     if hasattr(s, "pick"):           # Open3D ≥ 0.19
    #         hit = s.pick(cam, x, y)
    #         return hit.success, list(hit.world)

    #     # Open3D 0.17–0.18  →  cast_ray()
    #     hit = s.cast_ray(cam, x, y)
    #     if hit.geometry_id == -1:        # INVALID_ID in those releases
    #         return False, None
    #     return True, list(hit.world_coordinate)

    # def _make_world_ray(self, px, py):
    #     cam  = self._scene.scene.camera
    #     w, h = self._scene.frame.width, self._scene.frame.height

    #     ndc_x =  2.0 * px / w - 1.0
    #     ndc_y = -2.0 * py / h + 1.0   # flip Y

    #     p_near = np.array([ndc_x, ndc_y, -1.0, 1.0])
    #     p_far  = np.array([ndc_x, ndc_y,  1.0, 1.0])

    #     proj = np.asarray(cam.get_projection_matrix()).reshape(4, 4).T
    #     view = np.asarray(cam.get_view_matrix()).reshape(4, 4).T
    #     inv  = np.linalg.inv(proj @ view)

    #     w_near = inv @ p_near;  w_near /= w_near[3]
    #     w_far  = inv @ p_far ;  w_far  /= w_far[3]

    #     origin    = w_near[:3]
    #     direction = w_far[:3] - origin
    #     direction /= np.linalg.norm(direction)
    #     return origin, direction

    # def _ray_pick(self, px, py):
    #     o, d = self._make_world_ray(px, py)
    #     ray = o3d.core.Tensor([*o, *d], dtype=o3d.core.Dtype.Float32)

    #     ans = self._ray_scene.cast_ray(ray)

    #     if np.isinf(ans['t_hit'].numpy()[0]):   # miss
    #         return False, None

    #     t = float(ans['t_hit'])
    #     hit_pt = o + d * t
    #     return True, hit_pt.tolist()


    # def _on_mouse_event(self, event):
        
    #     if (self._pick_mode
    #             and event.type == gui.MouseEvent.Type.BUTTON_DOWN
    #             and event.is_button_down(gui.MouseButton.LEFT)):

    #         ok, p = self._ray_pick(event.x, event.y)
    #         if not ok:
    #             return gui.SceneWidget.EventCallbackResult.IGNORED
    #         else:              # hit something
    #             p = p                                            # world-space xyz
    #             if self._pick_mode == "tip":
    #                 # remove old visualisation (if any)
    #                 if self.tip_label:
    #                     self.remove_tip_point(self.tip_label)
    #                 self.tip_position = p
    #                 self.tip_label = self.add_tip_point(
    #                     p, "Tip Point")          # uses helper already in file

    #             elif self._pick_mode == "handle":
    #                 if self.handle_label:
    #                     self.remove_handle_point(self.handle_label)
    #                 self.handle_position = p
    #                 self.handle_label = self.add_handle_point(
    #                     p, "Handle Point")      # helper

    #             # downstream code (cube-coords, edits, …) may rely on the
    #             # updated positions, so run whatever you need here, e.g.:
    #             # self._edit_prebake()

    #             self._pick_mode = None          # leave pick mode
    #             return gui.SceneWidget.EventCallbackResult.CONSUMED

    #     # fall through – let default camera controls happen
    #     return gui.SceneWidget.EventCallbackResult.IGNORED
    
    
        # def _toggle_switch(self, name, state, item_type):
    #     old_state = self.switch_board[name]
    #     if old_state == state:
    #         return
    #     self.switch_board[name] = state

    #     if item_type == MESH_ITEM:
    #         if state:
    #             try:
    #                 self._scene.scene.add_model(name, self.name_to_geom[name])
    #             except:
    #                 self._scene.scene.add_geometry(
    #                     name, self.name_to_geom[name], self.settings.material)
    #             # --- also add to RaycastingScene ------------------------
    #             if name not in self._geom_to_ray_id:
    #                 tid = self._ray_scene.add_triangles(
    #                     o3d.t.geometry.TriangleMesh.from_legacy(
    #                         self.name_to_geom[name]))
    #                 self._geom_to_ray_id[name] = tid
    #             # -------------------------------------------------------
    #         else:
    #             self._scene.scene.remove_geometry(name)
    #             if name in self._geom_to_ray_id:
    #                 self._ray_scene.remove_geometry(self._geom_to_ray_id[name])
    #                 del self._geom_to_ray_id[name]

    #     elif item_type == OBB_ITEM:
    #         if state:
    #             self._scene.scene.add_geometry(
    #                 name, self.name_to_geom[name], self.settings.material)
    #             if name not in self._geom_to_ray_id:
    #                 tid = self._ray_scene.add_triangles(
    #                     o3d.t.geometry.TriangleMesh.from_legacy(
    #                         self.name_to_geom[name]))
    #                 self._geom_to_ray_id[name] = tid
    #         else:
    #             self._scene.scene.remove_geometry(name)
    #             if name in self._geom_to_ray_id:
    #                 self._ray_scene.remove_geometry(self._geom_to_ray_id[name])
    #                 del self._geom_to_ray_id[name]

    #     elif item_type == LABEL_ITEM:
    #         if state:
    #             label, center, _ = self.label_dict[name]
    #             obj = self._scene.add_3d_label(center, label)
    #             self.label_dict[name] = (label, center, obj)
    #         else:
    #             label, center, obj = self.label_dict[name]
    #             self._scene.remove_3d_label(obj)

    #     elif item_type == PC_ITEM:
    #         if state:
    #             self._scene.scene.add_geometry(
    #                 name, self.name_to_geom[name], self.settings.material)
    #             if name not in self._geom_to_ray_id:
    #                 tid = self._ray_scene.add_triangles(
    #                     o3d.t.geometry.TriangleMesh.from_legacy(
    #                         self.name_to_geom[name]))
    #                 self._geom_to_ray_id[name] = tid
    #         else:
    #             self._scene.scene.remove_geometry(name)
    #             if name in self._geom_to_ray_id:
    #                 self._ray_scene.remove_geometry(self._geom_to_ray_id[name])
    #                 del self._geom_to_ray_id[name]


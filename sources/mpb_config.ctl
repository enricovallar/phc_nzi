(set! num-bands 8)
(set! resolution 32)

(set! k-points (list (vector3 0.0 0.0 0.0) (vector3 0.5 0.0 0.0) (vector3 0.5 0.5 0.0) (vector3 0.0 0.0 0.0)))
(set! k-points (interpolate 4 k-points))
(set! geometry (list
    (make cylinder (center (vector3 0.0 0.0 0.0))(radius 0.2)(height 1e+20)(material (make dielectric (epsilon 12))))
))
(set! geometry-lattice (make lattice (size 1 1 no-size) (basis1  (vector3 1.0 0.0 0.0)) (basis2  (vector3 0.0 1.0 0.0)) ))
(run-te)

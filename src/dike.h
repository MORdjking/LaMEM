/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **   Project      : LaMEM
 **   License      : MIT, see LICENSE file for details
 **   Contributors : Anton Popov, Boris Kaus, see AUTHORS file for complete list
 **   Organization : Institute of Geosciences, Johannes-Gutenberg University, Mainz
 **   Contact      : kaus@uni-mainz.de, popov@uni-mainz.de
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/
//---------------------------------------------------------------------------                                                                                                      
//.................. MATERIAL PARAMETERS READING ROUTINES....................                                                                                                      
//---------------------------------------------------------------------------                                                                                                      
#ifndef __dike_h__
#define __dike_h__
//---------------------------------------------------------------------------   

struct FB; 
struct ConstEqCtx;
struct DBMat;
struct FDSTAG;
struct TSSol;
struct JacRes;
struct Controls;
struct AdvCtx; 

//---------------------------------------------------------------------------       
//.......................   Dike Parameters  .......................                                                                                                      
//---------------------------------------------------------------------------

struct Dike
{
public:
  PetscInt ID;        // dike ID
  PetscInt dyndike_start;  //starting timestep for dynamic diking if 0 then no dynamic diking

  PetscInt PhaseID, PhaseTransID, nPtr;      // associated material phase and phase transition IDs
  PetscInt istep_count, nD, j1, j2;
  PetscInt istep_nave;       //number of timesteps for time averaging
  PetscInt nstep_locate; // Locate dike every nstep_locate timestep to allow elastic stresses to settle down between relocations
  PetscInt out_stress;  //option to output mean stresses to std out
  PetscInt out_dikeloc;  //option to output dike location to std out
  PetscInt dike3D;  //option to use 3 dimensional diking (determined by xy plane stresses)
  PetscScalar Mf;        // amount of magma-accomodated extension in front of box 
  PetscScalar Mb;        // amount of magma-accommodated extension in back of box
  PetscScalar Mc;        // amount of magma-accommodated extension in center of box
  PetscScalar y_Mc;      // location in y direction of Mc, if in x-direction x_Mc needs to be given or in z-direction z_Mc
  PetscScalar x_Mc;
  PetscScalar z_Mc;
  PetscScalar T_brit;    // optional temperature of brittle-ductile transition for stress calculations
  PetscScalar T_dsol;    // optional temperature of dike solidus
  PetscScalar filtx; 
  PetscScalar filty;
  //PetscScalar ymindyn;
  //PetscScalar ymaxdyn;

  Vec sxx_eff_ave; // *delete after debug
  Vec sxx_eff_ave_hist; // *delete after debug
  Vec raw_sxx; // *delete after debug
  Vec raw_sxx_ave; // *delete after debug
  Vec raw_sxx_ave_hist; // *delete after debug
  Vec smooth_sxx; // *delete after debug
  Vec smooth_sxx_ave; // *delete after debug
  Vec smooth_sxx_ave_hist; // *delete after debug
  
  Vec focused_magPressure;
  Vec solidus;
  Vec magPresence;

  Vec hxx_ave;
  Vec hxx_ave_hist;
  Vec hxx_ave_smooth;
  Vec hyy_ave;
  Vec hyy_ave_hist;
  Vec hyy_ave_smooth;
  Vec sxx_ave;
  Vec sxx_ave_hist;
  Vec sxx_ave_smooth;
  Vec syy_ave;
  Vec syy_ave_hist;
  Vec syy_ave_smooth;
  Vec dxx_ave;
  Vec dxx_ave_hist;
  Vec dxx_ave_smooth;
  Vec dyy_ave;
  Vec dyy_ave_hist;
  Vec dyy_ave_smooth;
  Vec hP_ave;
  Vec hP_ave_hist;
  Vec hP_ave_smooth;
  Vec Pc_ave;
  Vec Pc_ave_hist;
  Vec Pc_ave_smooth;
  Vec lithP_ave;
  Vec lithP_ave_hist;
  Vec lithP_ave_smooth;
  Vec magPressure;
  Vec magPressure_hist;
  Vec magPressure_smooth;
  
  PetscScalar drhomagma;
  PetscScalar zmax_magma;
  PetscScalar magPfac;
  PetscScalar magPwidth;
  PetscScalar magPMeltFrac;

  PetscScalar A;      // Smoothing parameter for variable M calculation
	PetscScalar Ts;     // Tensile strength of rock for variable M calculation (Pa)
  PetscScalar zeta_0; // Initial bulk viscosity for variable M calculation (Pa*s)
  PetscScalar damp;   // damping parameter for variable M calculation
  PetscInt const_M;   // Flag to turn off var_M on per-dike basis
};
      
struct DBPropDike
{
  PetscInt numDike;                   // number of dikes
  Dike     matDike[_max_num_dike_];   // dike properties per dike ID
};

// create the dike strutures for read-in 
PetscErrorCode DBDikeCreate(DBPropDike *dbdike, DBMat *dbm, FB *fb, JacRes *jr, PetscBool PrintOutput);

// read in dike parameters
PetscErrorCode DBReadDike(DBPropDike *dbdike, DBMat *dbm, FB *fb, JacRes *jr, PetscBool PrintOutput);

// compute the added RHS of the dike for the continuity equation
PetscErrorCode GetDikeContr(JacRes *jr,                                                                                                                                
                            PetscScalar *phRat, // phase ratios in the control volume   
                            PetscInt &AirPhase,
                            PetscScalar &dikeRHS,
                            PetscScalar &y_c,
                            PetscInt J,
                            PetscScalar sxx_eff_ave_cell,
                            PetscScalar dx);

// compute dike heat after Behn & Ito, 2008
PetscErrorCode Dike_k_heatsource(JacRes *jr,
                                Material_t *phases,
                                PetscScalar &Tc,
                                PetscScalar *phRat, // phase ratios in the control volume
                                PetscScalar &k,
                                PetscScalar &rho_A,
                                PetscScalar &y_c,
                                PetscInt J,
                                PetscScalar sxx_eff_ave_cell); 

PetscErrorCode Locate_Dike_Zones(AdvCtx *actx, PetscInt sFlag);
PetscErrorCode Compute_sxx_magP(JacRes *jr, PetscInt nD, PetscInt sFlag);
PetscErrorCode Smooth_sxx_eff(JacRes *jr, PetscInt nD, PetscInt nPtr, PetscInt  j1, PetscInt j2, PetscInt sFlag);
PetscErrorCode Set_dike_zones(JacRes *jr, PetscInt nD, PetscInt nPtr, PetscInt  j1, PetscInt j2);
PetscErrorCode Set_dike_base(JacRes *jr, PetscInt nD, PetscInt nPtr, PetscInt  j1, PetscInt j2);
PetscErrorCode Compute_varDikingStress(JacRes *jr, PetscInt sFlag);
PetscErrorCode DynamicDike_ReadRestart(DBPropDike *dbdike, DBMat *dbm, JacRes *jr, TSSol *ts, FILE *fp);
PetscErrorCode DynamicDike_WriteRestart(JacRes *jr, FILE *fp);
PetscErrorCode DynamicDike_Destroy(JacRes *jr);

//---------------------------------------------------------------------------
#endif

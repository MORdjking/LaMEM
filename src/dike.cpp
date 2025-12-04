
/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **   Project      : LaMEM
 **   License      : MIT, see LICENSE file for details
 **   Contributors : Anton Popov, Boris Kaus, see AUTHORS file for complete list
 **   Organization : Institute of Geosciences, Johannes-Gutenberg University, Mainz
 **   Contact      : kaus@uni-mainz.de, popov@uni-mainz.de
 **
 ** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ @*/
/*

    This file defines properties for the dike which is defined as an additional 
    source term on the RHS of the continutiy equation

*/
//---------------------------------------------------------------------------
//.................. DIKE PARAMETERS READING ROUTINES....................
//---------------------------------------------------------------------------
#include "LaMEM.h"
#include "phase.h"
#include "parsing.h"
#include "JacRes.h"
#include "dike.h"
#include "constEq.h"
#include "bc.h"
#include "tssolve.h"
#include "scaling.h"
#include "fdstag.h"
#include "tools.h"
#include "surf.h"
#include "advect.h"

// added for debug file output *djking
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

//---------------------------------------------------------------------------
PetscErrorCode DBDikeCreate(DBPropDike *dbdike, DBMat *dbm, FB *fb, JacRes *jr, PetscBool PrintOutput)
{

	// read all dike parameter blocks from file
	Dike *dike;
	FDSTAG *fs;
	PetscScalar ***gsxx_eff_ave_hist, ***raw_gsxx_ave_hist, ***smooth_gsxx_ave_hist;
	PetscScalar ***ghxx_ave_hist, ***ghyy_ave_hist, ***gsxx_ave_hist; // *djking
	PetscScalar ***gsyy_ave_hist, ***gdxx_ave_hist, ***gdyy_ave_hist; // *djking
	PetscScalar ***ghP_ave_hist, ***gPc_ave_hist, ***glithP_ave_hist, ***gmagPressure_hist; // *djking
	PetscInt jj, nD, numDike, numdyndike, istep_nave;
	PetscInt i, j, istep_count, sx, sy, sisc, nx, ny;

	PetscFunctionBeginUser;

	if (!jr->ctrl.actDike)
		PetscFunctionReturn(0); // only execute this function if dikes are active

	fs = jr->fs;
	//===============
	// DIKE PARAMETER
	//===============

	// setup block access mode
	PetscCall(FBFindBlocks(fb, _OPTIONAL_, "<DikeStart>", "<DikeEnd>"));

	if (fb->nblocks)
	{
		// print overview of dike blocks from file
		if (PrintOutput)
			PetscPrintf(PETSC_COMM_WORLD, "Dike blocks : \n");

		// initialize ID for consistency checks

		for (jj = 0; jj < _max_num_dike_; jj++)
			dbdike->matDike[jj].ID = -1;

		// error checking
		if (fb->nblocks > _max_num_dike_)
			SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Too many dikes specified! Max allowed: %lld", (LLD)_max_num_dike_);

		// store actual number of dike blocks
		dbdike->numDike = fb->nblocks;

		if (PrintOutput)
			PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------------------------------\n");

		// read each individual dike block
		for (jj = 0; jj < fb->nblocks; jj++)
		{
			PetscCall(DBReadDike(dbdike, dbm, fb, jr, PrintOutput));
			fb->blockID++;
		}

		if (PrintOutput)
			PetscPrintf(PETSC_COMM_WORLD, "--------------------------------------------------------------------------\n");
	}

	PetscCall(FBFreeBlocks(fb));

	numdyndike = 0;

	numDike = dbdike->numDike;
	for (nD = 0; nD < numDike; nD++) // loop through all dike blocks
	{
		dike = dbdike->matDike + nD;
		if (dike->dyndike_start > 0 || jr->ctrl.var_M || jr->ctrl.sol_track)
		{
			numdyndike++;
			if (numdyndike == 1) //(take this out of this loop because it will be repeated with >1 dynamic dike)
			{
				// DM for 1D cell center vector. vector is ny+1 long, but for whatever reason that must appear in the first,
				// not second entry on line 2 below
				PetscCall(DMDACreate3dSetUp(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
											fs->dsy.tnods, fs->dsy.nproc, fs->dsz.nproc,
											fs->dsx.nproc, fs->dsy.nproc, fs->dsz.nproc, 1, 1,
											0, 0, 0, &jr->DA_CELL_1D));

				// DM for 2D cell center vector, with istep_nave planes for time averaging
				PetscCall(DMDACreate3dSetUp(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
											fs->dsx.tcels, fs->dsy.tcels, fs->dsz.nproc * dike->istep_nave,
											fs->dsx.nproc, fs->dsy.nproc, fs->dsz.nproc, 1, 1,
											0, 0, 0, &jr->DA_CELL_2D_tave));
			}


			// creating local vectors and inializing the history vector
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->magPresence));
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->solidus));
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->focused_magPressure));
			
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->sxx_eff_ave));
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->raw_sxx));
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->raw_sxx_ave));
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->smooth_sxx));
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->smooth_sxx_ave));
			
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->sxx_eff_ave_hist)); // *delete after debug
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->raw_sxx_ave_hist)); // *delete after debug
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->smooth_sxx_ave_hist)); // *delete after debug
			
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->sxx_eff_ave_hist, &gsxx_eff_ave_hist)); // *delete after debug
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->raw_sxx_ave_hist, &raw_gsxx_ave_hist)); // *delete after debug
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->smooth_sxx_ave_hist, &smooth_gsxx_ave_hist)); // *delete after debug
			
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->hxx_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->hyy_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->sxx_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->syy_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->dxx_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->dyy_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->hP_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->Pc_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->lithP_ave)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->magPressure)); // *djking
			
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->hxx_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->hyy_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->sxx_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->syy_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->dxx_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->dyy_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->hP_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->Pc_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->lithP_ave_hist)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D_tave, &dike->magPressure_hist)); // *djking
			
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->hxx_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->hyy_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->sxx_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->syy_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->dxx_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->dyy_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->hP_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->Pc_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->lithP_ave_smooth)); // *djking
			PetscCall(DMCreateLocalVector(jr->DA_CELL_2D, &dike->magPressure_smooth)); // *djking

			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->hxx_ave_hist, &ghxx_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->hyy_ave_hist, &ghyy_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->sxx_ave_hist, &gsxx_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->syy_ave_hist, &gsyy_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->dxx_ave_hist, &gdxx_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->dyy_ave_hist, &gdyy_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->hP_ave_hist, &ghP_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->Pc_ave_hist, &gPc_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->lithP_ave_hist, &glithP_ave_hist)); // *djking
			PetscCall(DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->magPressure_hist, &gmagPressure_hist)); // *djking
			
			PetscCall(DMDAGetCorners(jr->DA_CELL_2D_tave, &sx, &sy, &sisc, &nx, &ny, &istep_nave));
			
			for (j = sy; j < sy + ny; j++)
			
			{
				for (i = sx; i < sx + nx; i++)
				{
					for (istep_count = sisc; istep_count < sisc + istep_nave; istep_count++)
					{
						gsxx_eff_ave_hist[istep_count][j][i] = 0.0;
						raw_gsxx_ave_hist[istep_count][j][i] = 0.0;
						smooth_gsxx_ave_hist[istep_count][j][i] = 0.0;

						ghxx_ave_hist[istep_count][j][i] = 0.0; // *djking
						ghyy_ave_hist[istep_count][j][i] = 0.0; // *djking
						gsxx_ave_hist[istep_count][j][i] = 0.0; // *djking
						gsyy_ave_hist[istep_count][j][i] = 0.0; // *djking
						gdxx_ave_hist[istep_count][j][i] = 0.0; // *djking
						gdyy_ave_hist[istep_count][j][i] = 0.0; // *djking
						ghP_ave_hist[istep_count][j][i] = 0.0; // *djking
						gPc_ave_hist[istep_count][j][i] = 0.0; // *djking
						glithP_ave_hist[istep_count][j][i] = 0.0; // *djking
						gmagPressure_hist[istep_count][j][i] = 0.0; // *djking
					}
				}
			}
			
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->sxx_eff_ave_hist, &gsxx_eff_ave_hist));
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->raw_sxx_ave_hist, &raw_gsxx_ave_hist));
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->smooth_sxx_ave_hist, &smooth_gsxx_ave_hist));
			
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->hxx_ave_hist, &ghxx_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->hyy_ave_hist, &ghyy_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->sxx_ave_hist, &gsxx_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->syy_ave_hist, &gsyy_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->dxx_ave_hist, &gdxx_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->dyy_ave_hist, &gdyy_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->hP_ave_hist, &ghP_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->Pc_ave_hist, &gPc_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->lithP_ave_hist, &glithP_ave_hist)); // *djking
			PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->magPressure_hist, &gmagPressure_hist)); // *djking
		} // End if dyndike->start
	}	  // End loop through dikes

	PetscFunctionReturn(0);
}

//---------------------------------------------------------------------------
PetscErrorCode DBReadDike(DBPropDike *dbdike, DBMat *dbm, FB *fb, JacRes *jr, PetscBool PrintOutput)
{
	// read dike parameter from file
	Dike *dike;
	PetscInt ID;
	Scaling *scal;

	PetscFunctionBeginUser;

	// access context
	scal = dbm->scal;

	// Dike ID
	PetscCall(getIntParam(fb, _REQUIRED_, "ID", &ID, 1, dbdike->numDike - 1));
	fb->ID = ID;

	// get pointer to specified dike parameters
	dike = dbdike->matDike + ID;

	// check ID
	if (dike->ID != -1)
	{
		SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Duplicate of Dike option!");
	}

	// set ID
	dike->ID = ID;

	// set default value for Mc in case no Mc is provided
	dike->Mc = -1.0;
	// set default value for y_Mc in case it is not used (it does not matter since it is not accessed and checked for anywhere but might be better than not setting it)
	dike->y_Mc = 0.0;
	
	// read and store dike  parameters.
	PetscCall(getScalarParam(fb, _REQUIRED_, "Mf", &dike->Mf, 1, 1.0));
	PetscCall(getScalarParam(fb, _OPTIONAL_, "Mc", &dike->Mc, 1, 1.0));
	PetscCall(getScalarParam(fb, _REQUIRED_, "Mb", &dike->Mb, 1, 1.0));
	PetscCall(getScalarParam(fb, _OPTIONAL_, "y_Mc", &dike->y_Mc, 1, scal->length));
	PetscCall(getIntParam(fb, _REQUIRED_, "PhaseID", &dike->PhaseID, 1, dbm->numPhases - 1));
	PetscCall(getIntParam(fb, _REQUIRED_, "PhaseTransID", &dike->PhaseTransID, 1, dbm->numPhtr - 1));
	PetscCall(getIntParam(fb, _OPTIONAL_, "dyndike_start", &dike->dyndike_start, 1, -1));

	// variable M parameters
	if (jr->ctrl.var_M)
	{
		
		dike->A = 1e3;		 // default smoothing
		dike->Ts = 10e6;	 // default tensile strength
		dike->zeta_0 = 1e22; // default reference bulk viscosity
		dike->damp = 0;      // default damping value (0 percent)
		dike->const_M = 0;   // flag to turn off var_M (per-dike basis)
		dike->dike3D = 0;    // flag to turn on 3D diking (in xy-plane)
		
		PetscCall(getScalarParam(fb, _OPTIONAL_, "A", &dike->A, 1, 1));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "Ts", &dike->Ts, 1, 1));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "zeta_0", &dike->zeta_0, 1, 1));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "damp", &dike->damp, 1, 1));
		PetscCall(getIntParam(fb, _OPTIONAL_, "const_M", &dike->const_M, 1, 1));
		PetscCall(getIntParam(fb, _OPTIONAL_, "dike3D", &dike->dike3D, 1, 1));
		
		
		// scaling
		dike->A /= scal->stress_si;
		dike->Ts /= scal->stress_si;
		dike->zeta_0 /= scal->viscosity;
		dike->damp /= 100;
	}
	
	// parameters for average lithospheric stress calculations (includes magma pressure)
	if (dike->dyndike_start > 0 || jr->ctrl.var_M || jr->ctrl.sol_track) 
	{
		dike->T_dsol = 1000; // solids temperature (could connect to T_sol in future but would have to bring phase ratios in to do so)
		dike->T_brit = dike->T_dsol; // brittle-ductile transition temperature defaults to dike solidus if not user defined
		dike->zmax_magma = -15.0;
		dike->drhomagma = 500;
		dike->filtx = 1.5;
		dike->filty = 1.5;
		dike->istep_nave = 2;
		dike->nstep_locate = 1;
		dike->out_stress = 0;
		dike->out_dikeloc = 0;
		dike->magPfac = 1.0;
		dike->magPwidth = 1e+30;
		dike->magPMeltFrac = 1.0;
		// dike->ymindyn=-1e+30;
		// dike->ymaxdyn=1e+30;
		
		PetscCall(getScalarParam(fb, _OPTIONAL_, "T_dsol", &dike->T_dsol, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "T_brit", &dike->T_brit, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "filtx", &dike->filtx, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "filty", &dike->filty, 1, 1.0));
		
		PetscCall(getScalarParam(fb, _OPTIONAL_, "zmax_magma", &dike->zmax_magma, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "drhomagma", &dike->drhomagma, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "magPfac", &dike->magPfac, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "magPwidth", &dike->magPwidth, 1, 1.0));
		PetscCall(getScalarParam(fb, _OPTIONAL_, "magPMeltFrac", &dike->magPMeltFrac, 1, 1.0));
		// PetscCall(getScalarParam(fb, _OPTIONAL_, "ymindyn",	&dike->ymindyn,		1, 1.0));
		// PetscCall(getScalarParam(fb, _OPTIONAL_, "ymaxdyn",	&dike->ymaxdyn,		1, 1.0));
		
		PetscCall(getIntParam(fb, _OPTIONAL_, "istep_nave", &dike->istep_nave, 1, 50));
		PetscCall(getIntParam(fb, _OPTIONAL_, "nstep_locate", &dike->nstep_locate, 1, 1000));
		PetscCall(getIntParam(fb, _OPTIONAL_, "out_stress", &dike->out_stress, 1, 1));
		PetscCall(getIntParam(fb, _OPTIONAL_, "out_dikeloc", &dike->out_dikeloc, 1, 1));
		
		// scaling
		dike->filtx /= scal->length;
		dike->filty /= scal->length;
		dike->zmax_magma /= scal->length;
		dike->drhomagma /= scal->density;
		dike->magPwidth /= scal->length;
		dike->T_dsol = (dike->T_dsol + jr->scal->Tshift) / jr->scal->temperature;
		dike->T_brit = (dike->T_brit + jr->scal->Tshift) / jr->scal->temperature;

		// initialize so that when istep=dike_start, it is set to 0
		dike->istep_count = dike->istep_nave;
	}

	// print diking info to terminal
	if (PrintOutput)
	{
		PetscPrintf(PETSC_COMM_WORLD, "Dike [%lld]: ", (LLD)(dike->ID));
		if (dike->dyndike_start > 0)
		{
			PetscPrintf(PETSC_COMM_WORLD, "Dynamic starting at timestep = %lld\n", (LLD)(dike->dyndike_start));
		}
		else
		{
			PetscPrintf(PETSC_COMM_WORLD, "Static \n");
		}
		PetscPrintf(PETSC_COMM_WORLD, "   PhaseTransID=%lld PhaseID=%lld Mf=%g, Mb=%g, Mc=%g, y_Mc=%g \n",
					(LLD)(dike->PhaseTransID), (LLD)(dike->PhaseID), dike->Mf, dike->Mb, dike->Mc, dike->y_Mc);
		if (jr->ctrl.var_M && !(dike->const_M > 0))
		{
			PetscPrintf(PETSC_COMM_WORLD, "   Variable M option used:\n");
			PetscPrintf(PETSC_COMM_WORLD, "     A = %lld %s, Ts = %lld %s, zeta_0 = %g %s, damping = %.0f%%\n",
						(LLD)(dike->A * scal->stress_si), scal->lbl_stress_si, (LLD)(dike->Ts * scal->stress), scal->lbl_stress, dike->zeta_0 * scal->viscosity, scal->lbl_viscosity, dike->damp);
		}
		if (dike->dyndike_start > 0 || jr->ctrl.var_M)
		{
			PetscPrintf(PETSC_COMM_WORLD, "   gsxx_eff_ave smoothing parameters:\n");
			PetscPrintf(PETSC_COMM_WORLD, "     T_dsol = %1.0f %s, T_brit = %1.0f %s, filtx = %1.2f %s, filty = %1.2f %s\n",
						dike->T_dsol * scal->temperature - scal->Tshift, scal->lbl_temperature,
						dike->T_brit * scal->temperature - scal->Tshift, scal->lbl_temperature, 
						dike->filtx * scal->length, scal->lbl_length, 
						dike->filty * scal->length, scal->lbl_length);
			PetscPrintf(PETSC_COMM_WORLD, "     nstep_locate = %lld, istep_nave = %lld, istep_count = %lld\n",
						(LLD)(dike->nstep_locate), (LLD)(dike->istep_nave), (LLD)(dike->istep_count));
			PetscPrintf(PETSC_COMM_WORLD, "   excess magma pressure options:\n");
			PetscPrintf(PETSC_COMM_WORLD, "     zmax_magma = %1.0f %s, magPwidth = %.2e %s\n",
						dike->zmax_magma*scal->length, scal->lbl_length, dike->magPwidth*scal->length, scal->lbl_length);
			PetscPrintf(PETSC_COMM_WORLD, "     drhomagma = %1.0f %s, magPfac = %1.1f\n",
						dike->drhomagma*scal->density, scal->lbl_density, dike->magPfac);
		}

	}

	PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------------
PetscErrorCode GetDikeContr(JacRes *jr,
							PetscScalar *phRat, // phase ratios in the control volume
							PetscInt &AirPhase,
							PetscScalar &dikeRHS,
							PetscScalar &y_c,
							PetscInt J,
							PetscScalar sxx_eff_ave_cell,
							PetscScalar dx)

{

	BCCtx *bc;
	Dike *dike;
	Ph_trans_t *CurrPhTr;
	PetscInt i, nD, nPtr, numDike, numPhtr, nsegs;
	PetscScalar v_spread, M, left, right, front, back;
	PetscScalar y_distance, tempdikeRHS;
	PetscScalar P_comp, div_max, M_rat, zeta;

	PetscFunctionBeginUser;

	numDike = jr->dbdike->numDike; // number of dikes
	numPhtr = jr->dbm->numPhtr;
	bc = jr->bc;

	nPtr = 0;
	nD = 0;

	for (nPtr = 0; nPtr < numPhtr; nPtr++) // loop over all phase transitions blocks
	{

		// access the parameters of the phasetranstion block
		CurrPhTr = jr->dbm->matPhtr + nPtr;

		for (nD = 0; nD < numDike; nD++) // loop through all dike blocks
		{
			// access the parameters of the dike depending on the dike block
			dike = jr->dbdike->matDike + nD;

			// access the phase ID of the dike parameters of each dike
			i = dike->PhaseID;

			if (CurrPhTr->ID == dike->PhaseTransID) // compare the phaseTransID associated with the dike with the actual ID of the phase transition in this cell
			{
				// if in the dike zone
				if (phRat[i] > 0 && CurrPhTr->celly_xboundR[J] > CurrPhTr->celly_xboundL[J])
				{
					nsegs = CurrPhTr->nsegs;

					if (dike->Mb == dike->Mf && dike->Mc < 0.0) // spatially constant M
					{
						M = dike->Mf;
						v_spread = PetscAbs(bc->velin);
						left = CurrPhTr->celly_xboundL[J];
						right = CurrPhTr->celly_xboundR[J];

						if (jr->ctrl.var_M && !(dike->const_M > 0))
						{
							P_comp = sxx_eff_ave_cell - dike->Ts;

							// OG M dependent
							M_rat = M; // M ratio *revisit to include global var_M?
							// div_max = M_rat * 2 * (v_spread / (right - left)); // dike wide limit on divergence (right - left)
							div_max = M_rat * 2 * (v_spread / dx); // element wide limit rather than dike wide

							// PetscCall(PetscPrintf(PETSC_COMM_WORLD, "oldDivMax=%.4e, newDivMax=%.4e, dikeWidth=%.4e, elementWidth=%.4e\n", M_rat * 2 * (v_spread / (right - left)), M_rat * 2 * (v_spread / dx), (right - left), dx)); // *debug output *djking

/* 							// Strain rate dependent
							M_rat = M; // M ratio *revisit to include global var_M
							//sr_max = lithospheric_dxx_ave; // max strain rate (dxx in 2d) *revisit for 3d
							div_max = M_rat * sr_max_cell; */

							if (P_comp > 0) // diking occurs
							{
								// OG formulation
								zeta = dike->A * (dike->zeta_0 / P_comp) + P_comp / div_max;
								tempdikeRHS = P_comp / zeta;
								
/* 								// linear formulation
								zeta = dike->zeta_0;
								tempdikeRHS = PetscMin(P_comp / zeta, div_max); */
							}
							else // diking DOES NOT occur
							{
								tempdikeRHS = 0.0;
							}
						}
						else // not using var_M
						{
							tempdikeRHS = M * 2 * v_spread / (right - left);
						}
					}
					else if (dike->Mc >= 0.0) // Mf, Mc and Mb are all user defined
					{
						if (jr->ctrl.var_M && !(dike->const_M > 0)) // check variable M option isn't used
						{
							SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid option: var_M option requires uniform M");
						}

						left = CurrPhTr->celly_xboundL[J];
						right = CurrPhTr->celly_xboundR[J];
						front = CurrPhTr->ybounds[0];
						back = CurrPhTr->ybounds[2 * nsegs - 1];
						v_spread = PetscAbs(bc->velin);

						if (y_c >= dike->y_Mc)
						{
							// linear interpolation between different M values, Mc is M in the middle, acts as M in front, Mb is M in back
							y_distance = y_c - dike->y_Mc;
							M = dike->Mc + (dike->Mb - dike->Mc) * (y_distance / (back - dike->y_Mc));
							tempdikeRHS = M * 2 * v_spread / (right - left);
						}
						else
						{
							// linear interpolation between different M values, Mf is M in front, Mc acts as M in back
							y_distance = y_c - front;
							M = dike->Mf + (dike->Mc - dike->Mf) * (y_distance / (dike->y_Mc - front));
							tempdikeRHS = M * 2 * v_spread / (right - left);
						}
					}
					else if (dike->Mb != dike->Mf && dike->Mc < 0.0) // only Mf and Mb, they are different
					{
						if (jr->ctrl.var_M && !(dike->const_M > 0)) // check varaible M option isn't used
						{
							SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid option: var_M option requires uniform M");
						}

						left = CurrPhTr->celly_xboundL[J];
						right = CurrPhTr->celly_xboundR[J];
						front = CurrPhTr->ybounds[0];
						back = CurrPhTr->ybounds[2 * nsegs - 1];
						v_spread = PetscAbs(bc->velin);

						// linear interpolation between different M values, Mf is M in front, Mb is M in back
						y_distance = y_c - front;
						M = dike->Mf + (dike->Mb - dike->Mf) * (y_distance / (back - front));
						tempdikeRHS = M * 2 * v_spread / (right - left);
					}
					else // Mb and Mf don't exist (which should not occurr)
					{
						SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No values for Mb and Mf. Dike option invalid!");
					}

					// Divergence
					dikeRHS += (phRat[i] + phRat[AirPhase]) * tempdikeRHS; // Give full divergence if cell is part dike part air (*revisit Why??)

				} // close if phRat and xboundR>xboundL
			}	  // close phase transition and dike phase ID comparison
		}		  // close dike block loop
	}			  // close phase transition block loop
	PetscFunctionReturn(0);
}

//-----------------------------------------------------------------------------------------------------------------
PetscErrorCode Dike_k_heatsource(JacRes *jr,
								 Material_t *phases,
								 PetscScalar &Tc,
								 PetscScalar *phRat,
								 PetscScalar &k,
								 PetscScalar &rho_A,
								 PetscScalar &y_c,
								 PetscInt J,
								 PetscScalar hdiv_dike_cell)

{
	// parameters to determine dilation term
	BCCtx *bc;
	Dike *dike;
	Ph_trans_t *CurrPhTr;
	PetscInt i, nD, nPtr, numDike, numPhtr, nsegs;
	PetscScalar v_spread, M, left, right, front, back;
	PetscScalar y_distance, tempdikeRHS;
	// PetscScalar P_comp, div_max, M_rat, zeta; version prior to hdiv_dike_cell *djking

	// heating parameters
	Material_t *mat;
	PetscScalar kfac;

	PetscFunctionBeginUser;

	numDike = jr->dbdike->numDike; // number of dikes
	numPhtr = jr->dbm->numPhtr;
	bc = jr->bc;

	nPtr = 0;
	nD = 0;
	kfac = 0;

	for (nPtr = 0; nPtr < numPhtr; nPtr++) // loop over all phase transitions blocks
	{

		// access the parameters of the phasetranstion block
		CurrPhTr = jr->dbm->matPhtr + nPtr;

		for (nD = 0; nD < numDike; nD++) // loop through all dike blocks
		{
			// access the parameters of the dike depending on the dike block
			dike = jr->dbdike->matDike + nD;

			// access the phase ID of the dike parameters of each dike
			i = dike->PhaseID;

			if (CurrPhTr->ID == dike->PhaseTransID) // compare the phaseTransID associated with the dike with the actual ID of the phase transition in this cell
			{
				// if in the dike zone
				if (phRat[i] > 0 && CurrPhTr->celly_xboundR[J] > CurrPhTr->celly_xboundL[J])
				{
					nsegs = CurrPhTr->nsegs;

					if (dike->Mb == dike->Mf && dike->Mc < 0.0) // spatially constant M
					{
						M = dike->Mf;
						v_spread = PetscAbs(bc->velin);
						left = CurrPhTr->celly_xboundL[J];
						right = CurrPhTr->celly_xboundR[J];

						if (jr->ctrl.var_M && !(dike->const_M > 0))
						{
							// dependent on history diking (this may miss some heating if dike zone moves outside of diking blocks)
							tempdikeRHS = hdiv_dike_cell;

							// changes for each iteration
/* 							P_comp = sxx_eff_ave_cell - dike->Ts;
							M_rat = M; // M ratio *revisit
							div_max = M_rat * 2 * (v_spread / (right - left));

							if (P_comp > 0) // diking occurs
							{
								zeta = dike->A * (dike->zeta_0 / P_comp) + P_comp / div_max;
								tempdikeRHS = P_comp / zeta;
							}
							else // diking DOES NOT occur
							{
								tempdikeRHS = 0.0;
							} */
						}
						else // not using var_M
						{
							tempdikeRHS = M * 2 * v_spread / (right - left);
						}
					}
					else if (dike->Mc >= 0.0) // Mf, Mc and Mb are all user defined
					{
						if (jr->ctrl.var_M && !(dike->const_M > 0)) // check varaible M option isn't used
						{
							SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid option: var_M option requires single global M (i.e. Mf = Mb)");
						}

						left = CurrPhTr->celly_xboundL[J];
						right = CurrPhTr->celly_xboundR[J];
						front = CurrPhTr->ybounds[0];
						back = CurrPhTr->ybounds[2 * nsegs - 1];
						v_spread = PetscAbs(bc->velin);

						if (y_c >= dike->y_Mc)
						{
							// linear interpolation between different M values, Mc is M in the middle, acts as M in front, Mb is M in back
							y_distance = y_c - dike->y_Mc;
							M = dike->Mc + (dike->Mb - dike->Mc) * (y_distance / (back - dike->y_Mc));
							tempdikeRHS = M * 2 * v_spread / (right - left);
						}
						else
						{
							// linear interpolation between different M values, Mf is M in front, Mc acts as M in back
							y_distance = y_c - front;
							M = dike->Mf + (dike->Mc - dike->Mf) * (y_distance / (dike->y_Mc - front));
							tempdikeRHS = M * 2 * v_spread / (right - left);
						}
					}
					else if (dike->Mb != dike->Mf && dike->Mc < 0.0) // only Mf and Mb, they are different
					{
						if (jr->ctrl.var_M && !(dike->const_M > 0)) // check varaible M option isn't used
						{
							SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid option: var_M option requires single global M (i.e. Mf = Mb)");
						}

						left = CurrPhTr->celly_xboundL[J];
						right = CurrPhTr->celly_xboundR[J];
						front = CurrPhTr->ybounds[0];
						back = CurrPhTr->ybounds[2 * nsegs - 1];
						v_spread = PetscAbs(bc->velin);

						// linear interpolation between different M values, Mf is M in front, Mb is M in back
						y_distance = y_c - front;
						M = dike->Mf + (dike->Mb - dike->Mf) * (y_distance / (back - front));
						tempdikeRHS = M * 2 * v_spread / (right - left);
					}
					else // Mb and Mf don't exist (which should not occurr)
					{
						SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No values for Mb and Mf. Dike option invalid!");
					}

					mat = &phases[i];

					// adjust k and heat source according to Behn & Ito [2008]
					if (Tc < mat->T_liq && Tc > mat->T_sol) // partially molten state
					{
						kfac += phRat[i] / (1 + (mat->Latent_hx / (mat->Cp * (mat->T_liq - mat->T_sol))));
						rho_A += phRat[i] * (mat->rho * mat->Cp) * (mat->T_liq - Tc) * tempdikeRHS; // Cp*rho not used in the paper, added to conserve units of rho_A
					}
					else if (Tc <= mat->T_sol) // solid state
					{
						kfac += phRat[i];
						rho_A += phRat[i] * (mat->rho * mat->Cp) * ((mat->T_liq - Tc) + mat->Latent_hx / mat->Cp) * tempdikeRHS;
					}
					else if (Tc >= mat->T_liq) // liquid state
					{
						kfac += phRat[i];
					}
					// end adjust k and heat source according to Behn & Ito [2008]

					k = kfac * k;

				} // end if phRat and xboundR>xboundL
			}	  // close phase transition and phase ID comparison
		}		  // end dike block loop
	}			  // close phase transition block loop

	PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------------
PetscErrorCode Locate_Dike_Zones(AdvCtx *actx, PetscInt sFlag)
{

	Controls *ctrl;
	JacRes *jr;
	Dike *dike;
	Ph_trans_t *CurrPhTr;
	FDSTAG *fs;
	PetscInt nD, numDike, numPhtr, nPtr, n; //, icounter; *djking
	PetscInt j, j1, j2, sx, sy, sz, ny, nx, nz;
	PetscErrorCode ierr;

	PetscFunctionBeginUser;

	jr = actx->jr;
	fs = jr->fs;
	ctrl = &jr->ctrl;
	
	if (!ctrl->actDike || jr->ts->istep + 1 == 0) PetscFunctionReturn(0); // only execute if diking is activated
	// if (!ctrl->actDike || !ctrl->sol_track || jr->ts->istep + 1 == 0) PetscFunctionReturn(0); // Solidus tracking outside of diking?? debugging

	// copy prior step diking values and save as a history diking value for dike heating (used with var_M) *djking
	ierr = VecCopy(jr->dc, jr->hdc); CHKERRQ(ierr);
	
	PetscPrintf(PETSC_COMM_WORLD, "\n");
	numDike = jr->dbdike->numDike; // number of dikes
	numPhtr = jr->dbm->numPhtr;

/* 	icounter = 0; // not needed *djking */
	ierr = DMDAGetCorners(fs->DA_CEN, &sx, &sy, &sz, &nx, &ny, &nz); CHKERRQ(ierr);

	if (ctrl->actDike)
	{
		for (nD = 0; nD < numDike; nD++)
		{
			// access the parameters of the dike depending on the dike block
			dike = jr->dbdike->matDike + nD;

			// if there is any reason to find stress, magmatic pressure, or even the solidus
			if (dike->dyndike_start > 0 || jr->ctrl.var_M || jr->ctrl.sol_track)
			{
/* 				// compute lithostatic pressure [this has aleady been done at the end of initial guess and solve steps *djking]
				if (icounter == 0)
				{
					ierr = JacResGetLithoStaticPressure(jr); CHKERRQ(ierr);
					ierr = ADVInterpMarkToCell(actx); CHKERRQ(ierr);
				}
				icounter++; */

				//---------------------------------------------------------------------------------------------
				//  Find dike phase transition
				//---------------------------------------------------------------------------------------------
				nPtr = -1;
				for (n = 0; n < numPhtr; n++)
				{
					CurrPhTr = jr->dbm->matPhtr + n;
					if (CurrPhTr->ID == dike->PhaseTransID)
					{
						nPtr = n;
					}
				} // end loop over Phtr

				if (nPtr == -1)
					SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "PhaseTransID problems with dike %lld, nPtr=%lld\n", (LLD)(nD), (LLD)(nPtr));

				CurrPhTr = jr->dbm->matPhtr + nPtr;

				//---------------------------------------------------------------------------------------------
				//  Find y-bounds of current dynamic dike
				//---------------------------------------------------------------------------------------------
				j1 = ny - 1;
				j2 = 0;
				for (j = 0; j < ny; j++)
				{
					if (CurrPhTr->celly_xboundR[j] > CurrPhTr->celly_xboundL[j])
					{
						j1 = (PetscInt)min(j1, j);
						j2 = (PetscInt)max(j2, j);
					}
				}

				ierr = Compute_sxx_magP(jr, nD, sFlag); CHKERRQ(ierr); // compute mean effective sxx across the lithosphere

				ierr = Smooth_sxx_eff(jr, nD, nPtr, j1, j2, sFlag); CHKERRQ(ierr); // smooth mean effective sxx

				// Only relocate dike zone if dynamic diking is on and if on an nstep_locate timestep
				if (dike->dyndike_start > 0 && (jr->ts->istep + 1 >= dike->dyndike_start) && ((jr->ts->istep + 1) % dike->nstep_locate) == 0)
				{
					PetscPrintf(PETSC_COMM_WORLD, "Locating Dike zone: istep=%lld dike # %lld\n", (LLD)(jr->ts->istep + 1), (LLD)(nD));
					ierr = Set_dike_zones(jr, nD, nPtr, j1, j2); CHKERRQ(ierr); // centered on peak sxx_eff_ave
				}

				// change z boundary of dike box if trackSolidus is set
				if (jr->ctrl.sol_track)
				{
					ierr = Set_dike_base(jr, nD, nPtr, j1, j2); CHKERRQ(ierr); // use solidus values to set zbound of dike
				}
			}
		}
	}
/*  // Solidus tracking outside of diking?? debugging
	// Currently requires dike block params b/c calculates more than just solidus
	
	else if (jr->ctrl.sol_track) // gets solidus array (as well as average sxx, etc...)
	{
		nD = 0;
		ierr = Compute_sxx_magP(jr, nD); CHKERRQ(ierr); // compute mean effective sxx across the lithosphere
	} */
	
	PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------------
PetscErrorCode Compute_sxx_magP(JacRes *jr, PetscInt nD, PetscInt sFlag)
{
  MPI_Request srequest, rrequest;

  Vec         vhxx, vhyy, vsxx, vsyy, vdxx, vdyy;
  Vec         vhP, vPc, vlithP, vliththick;
  Vec         vcdxx, vcdyy, vzsol;
  
  PetscScalar ***hxx, ***hyy, ***sxx, ***syy, ***dxx, ***dyy;
  PetscScalar ***hP, ***Pc, ***lithP, ***liththick;
  PetscScalar ***cdxx, ***cdyy, ***zsol;
  
  PetscScalar *lhxx, *lhyy, *lsxx, *lsyy, *ldxx, *ldyy;
  PetscScalar *lhP, *lPc, *llithP, *lliththick;
  PetscScalar *lcdxx, *lcdyy, *lzsol;
  
  PetscScalar ***ghxx_ave, ***ghyy_ave, ***gsxx_ave, ***gsyy_ave;
  PetscScalar ***gdxx_ave, ***gdyy_ave, ***ghP_ave, ***gPc_ave, ***glithP_ave;
  
  PetscScalar x_c, y_c, z_c;
  PetscInt istep, nstep_out, iteration;
  
  PetscScalar ***solidus, ***magPresence;
  PetscScalar ***gmagPressure, ***focused_magPressure;
  PetscScalar zsol_max_local = -PETSC_MAX_REAL,  zsol_max_global;
  PetscScalar dz, ***lT, ***lp, ***p_lith, Tc, *grav, magP, magma_presence;
  PetscInt    i, j, k, sx, sy, sz, nx, ny, nz, L, ID, AirPhase;
  
  Vec         vogsxx; // *delete after debug
  PetscScalar ***ogsxx, *logsxx; // *delete after debug
  PetscScalar ***gsxx_eff_ave; // *delete after debug
  PetscScalar ***raw_gsxx, ***smooth_gsxx; // *delete after debug
  PetscScalar ***raw_gsxx_ave, ***smooth_gsxx_ave; // *delete after debug
  
  Vec         vbritthick; // *djking
  PetscScalar ***britthick; // *djking
  PetscScalar *lbritthick;
  PetscScalar T_dsol, T_brit; // *djking


  PetscMPIInt rank;


  FDSTAG      *fs;
  Dike        *dike;
  Discret1D   *dsz;
  SolVarCell  *svCell;
  Controls    *ctrl;

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ctrl = &jr->ctrl;
  grav = ctrl->grav;

  fs  =  jr->fs;
  dsz = &fs->dsz;
  L   =  (PetscInt)dsz->rank;
  AirPhase  = jr->surf->AirPhase;

  istep=jr->ts->istep+1; // *djking
  nstep_out=jr->ts->nstep_out; // *djking
  iteration = jr->ts->itNum; // *djking

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  //Access arrays
  ierr = DMDAVecGetArray(fs->DA_CEN, jr->lT,   &lT);  CHKERRQ(ierr);
  ierr = DMDAVecGetArray(fs->DA_CEN, jr->lp,   &lp);  CHKERRQ(ierr);
  ierr = DMDAVecGetArray(fs->DA_CEN, jr->lp_lith, &p_lith); CHKERRQ(ierr);

  dike = jr->dbdike->matDike+nD;

  // get local grid sizes
  ierr = DMDAGetCorners(fs->DA_CEN, &sx, &sy, &sz, &nx, &ny, &nz); CHKERRQ(ierr);

  // get integration/communication buffer (Gets a PETSc vector, vsxx, that may be used with the DM global routines)
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhxx); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhyy); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsyy); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdxx); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdyy); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhP); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vPc); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vlithP); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vliththick); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vcdxx); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vcdyy); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vzsol); CHKERRQ(ierr);
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vogsxx); CHKERRQ(ierr); // *delete after debug
  ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vbritthick); CHKERRQ(ierr); // *djking

  ierr = VecZeroEntries(vhxx); CHKERRQ(ierr);
  ierr = VecZeroEntries(vhyy); CHKERRQ(ierr);
  ierr = VecZeroEntries(vsxx); CHKERRQ(ierr);
  ierr = VecZeroEntries(vsyy); CHKERRQ(ierr);
  ierr = VecZeroEntries(vdxx); CHKERRQ(ierr);
  ierr = VecZeroEntries(vdyy); CHKERRQ(ierr);
  ierr = VecZeroEntries(vPc); CHKERRQ(ierr);
  ierr = VecZeroEntries(vhP); CHKERRQ(ierr);
  ierr = VecZeroEntries(vlithP); CHKERRQ(ierr);
  ierr = VecZeroEntries(vliththick); CHKERRQ(ierr);
  ierr = VecZeroEntries(vcdxx); CHKERRQ(ierr);
  ierr = VecZeroEntries(vcdyy); CHKERRQ(ierr);
  ierr = VecZeroEntries(vzsol); CHKERRQ(ierr);
  ierr = VecZeroEntries(vogsxx); CHKERRQ(ierr); // *delete after debug
  ierr = VecZeroEntries(vbritthick); CHKERRQ(ierr); // *djking

  // open index buffer for computation (sxx the array that shares data with vector vsxx and is indexed with global dimensions<<G.Ito)
  // DMDAVecGetArray(DM da,Vec vec,void *array) Returns a multiple dimension array that shares data with the underlying vector 
  // and is indexed using the global dimension
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhxx, &hxx); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhyy, &hyy); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx, &sxx); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsyy, &syy); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdxx, &dxx); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdyy, &dyy); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhP, &hP); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vPc, &Pc); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vlithP, &lithP); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vliththick, &liththick); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vcdxx, &cdxx); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vcdyy, &cdyy); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vzsol, &zsol); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vogsxx, &ogsxx); CHKERRQ(ierr); // *delete after debug
  ierr = DMDAVecGetArray(jr->DA_CELL_2D, vbritthick, &britthick); CHKERRQ(ierr); // *djking

  // open linear buffer for send/receive  (returns the point, lsxx, that contains this processor portion of vector data, vsxx<<G.Ito)
  //Returns a pointer to a contiguous array that contains this processors portion of the vector data.
  ierr = VecGetArray(vhxx, &lhxx); CHKERRQ(ierr);
  ierr = VecGetArray(vhyy, &lhyy); CHKERRQ(ierr);
  ierr = VecGetArray(vsxx, &lsxx); CHKERRQ(ierr);
  ierr = VecGetArray(vsyy, &lsyy); CHKERRQ(ierr);
  ierr = VecGetArray(vdxx, &ldxx); CHKERRQ(ierr);
  ierr = VecGetArray(vdyy, &ldyy); CHKERRQ(ierr);
  ierr = VecGetArray(vhP, &lhP); CHKERRQ(ierr);
  ierr = VecGetArray(vPc, &lPc); CHKERRQ(ierr);
  ierr = VecGetArray(vlithP, &llithP); CHKERRQ(ierr);
  ierr = VecGetArray(vliththick, &lliththick); CHKERRQ(ierr);
  ierr = VecGetArray(vcdxx, &lcdxx); CHKERRQ(ierr);
  ierr = VecGetArray(vcdyy, &lcdyy); CHKERRQ(ierr);
  ierr = VecGetArray(vzsol, &lzsol); CHKERRQ(ierr);
  ierr = VecGetArray(vogsxx, &logsxx); CHKERRQ(ierr); // *delete after debug
  ierr = VecGetArray(vbritthick, &lbritthick); CHKERRQ(ierr); // *djking


  // receive from top domain (next)  dsz->grnext is the next proc up (in increasing z). Top to bottom doesn't matter here, its this way
  // because the code is patterned after GetLithoStaticPressure
  if(dsz->nproc != 1 && dsz->grnext != -1)
  {
     ierr = MPI_Irecv(lhxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 ierr = MPI_Irecv(lhyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 ierr = MPI_Irecv(lsyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(ldxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 ierr = MPI_Irecv(ldyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Irecv(lhP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 ierr = MPI_Irecv(lPc, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Irecv(llithP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lliththick, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Irecv(lcdxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 ierr = MPI_Irecv(lcdyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
	 ierr = MPI_Irecv(lzsol, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
	 ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
	 ierr = MPI_Irecv(logsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr); // *delete after debug
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr); // *delete after debug

     ierr = MPI_Irecv(lbritthick, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
  }

  ///// INTEGRATE LITHSPHERIC SXX, LITHOSTATIC PRESSURE (lithP), AND THICKNESS OF T-DETERMINED LITHOSPHERE/////
  T_brit = dike->T_brit; // brittle/ductile transition temperature
  T_dsol = dike->T_dsol; // dike solidus isotherm
  for (k = sz + nz - 1; k >= sz; k--)
  {
	  dz = SIZE_CELL(k, sz, (*dsz));
	  z_c = COORD_CELL(k, sz, fs->dsz);

	  START_PLANE_LOOP

	  GET_CELL_ID(ID, i - sx, j - sy, k - sz, nx, ny); // GET_CELL_ID needs local indices
	  svCell = &jr->svCell[ID];
	  Tc = lT[k][j][i];
	  x_c = COORD_CELL(i, sx, fs->dsx);
	  y_c = COORD_CELL(j, sy, fs->dsy);
	  if ((Tc <= T_dsol) && (svCell->phRat[AirPhase] < 1.0))
	  {
		  dz = SIZE_CELL(k, sz, (*dsz));

		  // Presure calculated from Solidus
		  hP[L][j][i] += svCell->svBulk.pn * dz;  // integrating dz-weighted history pressure
		  Pc[L][j][i] += lp[k][j][i] * dz;		  // integrating dz-weighted pressure
		  lithP[L][j][i] += p_lith[k][j][i] * dz; // integrating lithostatic pressure
		  liththick[L][j][i] += dz;				  // integrating thickness
		  
		  if ((Tc <= T_brit) && (svCell->phRat[AirPhase] < 1.0))
		  {
			  hxx[L][j][i] += svCell->hxx * dz; // integrating dz-weighted deviatoric history stress (x-direction)
			  hyy[L][j][i] += svCell->hyy * dz; // integrating dz-weighted deviatoric history stress (y-direction)
			  sxx[L][j][i] += svCell->sxx * dz; // integrating dz-weighted deviatoric stress
			  syy[L][j][i] += svCell->syy * dz; // integrating dz-weighted deviatoric stress
			  dxx[L][j][i] += svCell->dxx * dz; // integrating dz-weighted deviatoric strain rate
			  dyy[L][j][i] += svCell->dyy * dz; // integrating dz-weighted deviatoric strain rate

			  cdxx[L][j][i] += (svCell->sxx / (2 * svCell->svDev.eta)) * dz; // integrating dz-weighted deviatoric strain rate from stress
			  cdyy[L][j][i] += (svCell->syy / (2 * svCell->svDev.eta)) * dz; // integrating dz-weighted deviatoric strain rate from stress

			  britthick[L][j][i] += dz; // integrating brittle layer thickness

			  if (sFlag == 1) // during dike locate step *djking
			  {
				  ogsxx[L][j][i] += (svCell->hxx - svCell->svBulk.pn) * dz; // integrating dz-weighted total history stress
			  }
			  else // within picard iterations
			  {
				  if (iteration == 0)
				  {
					  ogsxx[L][j][i] += (svCell->hxx - svCell->svBulk.pn) * dz; // integrating dz-weighted total history stress
				  }
				  else
				  {
					  ogsxx[L][j][i] += (svCell->sxx - lp[k][j][i]) * dz; // integrating dz-weighted total history stress
				  }
			  }

			  // OG formulas...
			  // sxx[L][j][i]+=(svCell->hxx - svCell->svBulk.pn)*dz;  //integrating dz-weighted total history stress
			  // sxx[L][j][i]+=svCell->hxx*dz;  //integrating dz-weighted deviatoric stress *djking
			  // sxx[L][j][i]+=(svCell->sxx - lp[L][j][i])*dz;  //integrating dz-weighted total stress
			  // sxx[L][j][i]+=(svCell->sxx)*dz;  //integrating dz-weighted total stress

			  if (x_c < 0.3 && x_c > 0.0 && y_c == -1.5 && z_c < -3 && z_c > -3.3) // || ID==82702) *debugging
			  {
				  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sFlag=%d, x_c=%.2f, y_c=%.2f, z_c=%.2f, hxx=%.4e, hyy=%.4e, sxx=%.4e, syy=%.4e, dxx=%.4e, dyy=%.4e, hP=%.4e, Pc=%.4e, lithP=%.4e, cdxx=%.4e, cdyy=%.4e, eta=%.4e\n", sFlag, x_c, y_c, z_c, svCell->hxx, svCell->hyy, svCell->sxx, svCell->syy, svCell->dxx, svCell->dyy, svCell->svBulk.pn, lp[k][j][i], p_lith[k][j][i], svCell->sxx / (2 * svCell->svDev.eta), svCell->syy / (2 * svCell->svDev.eta), svCell->svDev.eta));
			  }
			  /* 			if (x_c<6.3 && x_c>6.0 && y_c==-1.5 && z_c<-3 && z_c>-3.3) // || ID==82702) *debugging
						  {
							  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sFlag=%d, x_c=%.2f, y_c=%.2f, z_c=%.2f, hxx=%.4e, hyy=%.4e, sxx=%.4e, syy=%.4e, dxx=%.4e, dyy=%.4e, hP=%.4e, Pc=%.4e, lithP=%.4e, cdxx=%.4e, cdyy=%.4e, eta=%.4e\n", sFlag, x_c, y_c, z_c, svCell->hxx, svCell->hyy, svCell->sxx, svCell->syy, svCell->dxx, svCell->dyy, svCell->svBulk.pn, lp[k][j][i], p_lith[k][j][i], svCell->sxx/(2*svCell->svDev.eta), svCell->syy/(2*svCell->svDev.eta), svCell->svDev.eta));
						  }
						  if (x_c<36.3 && x_c>36.0 && y_c==-1.5 && z_c<-3 && z_c>-3.3) // || ID==82702) *debugging
						  {
							  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sFlag=%d, x_c=%.2f, y_c=%.2f, z_c=%.2f, hxx=%.4e, hyy=%.4e, sxx=%.4e, syy=%.4e, dxx=%.4e, dyy=%.4e, hP=%.4e, Pc=%.4e, lithP=%.4e, cdxx=%.4e, cdyy=%.4e, eta=%.4e\n", sFlag, x_c, y_c, z_c, svCell->hxx, svCell->hyy, svCell->sxx, svCell->syy, svCell->dxx, svCell->dyy, svCell->svBulk.pn, lp[k][j][i], p_lith[k][j][i], svCell->sxx/(2*svCell->svDev.eta), svCell->syy/(2*svCell->svDev.eta), svCell->svDev.eta));
						  } */
		  }

		  /* 		PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sFlag=%d, x_c=%.2f, y_c=%.2f, z_c=%.2f, hxx=%.4e, hyy=%.4e, sxx=%.4e, syy=%.4e, dxx=%.4e, dyy=%.4e, hP=%.4e, Pc=%.4e, lithP=%.4e, cdxx=%.4e, cdyy=%.4e, eta=%.4e\n", sFlag, x_c, y_c, z_c, svCell->hxx, svCell->hyy, svCell->sxx, svCell->syy, svCell->dxx, svCell->dyy, svCell->svBulk.pn, lp[k][j][i], p_lith[k][j][i], svCell->sxx/(2*svCell->svDev.eta), svCell->syy/(2*svCell->svDev.eta), svCell->svDev.eta)); */
	  }

	  // interpolate depth to the diking solidus
	  if ((Tc <= T_dsol) && (T_dsol < lT[k - 1][j][i]))
	  {
		  zsol[L][j][i] = dsz->ccoor[k - sz] + (dsz->ccoor[k - sz - 1] - dsz->ccoor[k - sz]) / (lT[k - 1][j][i] - Tc) * (T_dsol - Tc);
	  }

	  END_PLANE_LOOP
  }

  // After integrating thickness and dz-weighted total stress, send it down to the next proc.
  if (dsz->nproc != 1 && dsz->grprev != -1)
  {
	  ierr = MPI_Isend(lhxx, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(lhyy, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(lsxx, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(lsyy, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(ldxx, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(ldyy, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);

	  ierr = MPI_Isend(lhP, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(lPc, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);

	  ierr = MPI_Isend(llithP, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(lliththick, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  
	  ierr = MPI_Isend(lcdxx, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  ierr = MPI_Isend(lcdyy, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  
	  ierr = MPI_Isend(lzsol, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr);
	  
	  ierr = MPI_Isend(logsxx, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr); // *delete after debug
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr); // *delete after debug

	  ierr = MPI_Isend(lbritthick, (PetscMPIInt)(nx * ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr); // *djking
	  ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE); CHKERRQ(ierr); // *djking
	}

	//Now receive/send the answer from successive previous (underlying) procs so all procs have the answers
  if(dsz->nproc != 1 && dsz->grprev != -1)
  {
     ierr = MPI_Irecv(lhxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lhyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lsyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(ldxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(ldyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Irecv(lhP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lPc, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Irecv(llithP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Irecv(lliththick, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
	 ierr = MPI_Irecv(lcdxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
	 ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 ierr = MPI_Irecv(lcdyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
	 ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Irecv(lzsol, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     
	 ierr = MPI_Irecv(logsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr); // *delete after debug
     ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr); // *delete after debug

	 ierr = MPI_Irecv(lbritthick, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr); // *djking
	 ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr); // *djking
	}

  if(dsz->nproc != 1 && dsz->grnext != -1)
  {
     ierr = MPI_Isend(lhxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(lhyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(lsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(lsyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(ldxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(ldyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Isend(lhP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(lPc, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Isend(llithP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(lliththick, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Isend(lcdxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     ierr = MPI_Isend(lcdyy, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	 
     ierr = MPI_Isend(lzsol, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
     
	 ierr = MPI_Isend(logsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr); // *delete after debug
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr); // *delete after debug

	 ierr = MPI_Isend(lbritthick, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsz->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr); // *djking
     ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr); // *djking
  }

  //now all cores in z have the same solution so give that to the stress array
  
  ///// CALCULATE AVERAGE LITHOSPHERIC VALUES /////

  // (gdev is the array that shares data with devxx_mean and is indexed with global dimensions)
  
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->sxx_eff_ave, &gsxx_eff_ave); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->raw_sxx, &raw_gsxx); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->raw_sxx_ave, &raw_gsxx_ave); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->smooth_sxx, &smooth_gsxx); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->smooth_sxx_ave, &smooth_gsxx_ave); CHKERRQ(ierr); // *delete after debug
  
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->magPresence, &magPresence); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->solidus, &solidus); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->focused_magPressure, &focused_magPressure); CHKERRQ(ierr);
	
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hxx_ave, &ghxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hyy_ave, &ghyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->sxx_ave, &gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->syy_ave, &gsyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->dxx_ave, &gdxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->dyy_ave, &gdyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hP_ave, &ghP_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->Pc_ave, &gPc_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->lithP_ave, &glithP_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->magPressure, &gmagPressure); CHKERRQ(ierr);

	// store solidus in dike structure and find solidus max for solidus tracking (during locate dike step only)
	if (sFlag == 1) // *djking
	{
		START_PLANE_LOOP
		solidus[L][j][i] = zsol[L][j][i];						  // store zsol in the solidus array outside function
		zsol_max_local = PetscMax(zsol[L][j][i], zsol_max_local); // finding local max solidus (thinnest lithosphere)
		END_PLANE_LOOP
		MPI_Allreduce(&zsol_max_local, &zsol_max_global, 1, MPIU_SCALAR, MPI_MAX, PETSC_COMM_WORLD); // find solidus global max
	}

/* 	// to find 2 peaks if we switch to using focused_magPressure *djking testing
	PetscScalar zsol_max_local[2] = {-PETSC_MAX_REAL, -PETSC_MAX_REAL}; // Array to store local top two maxima
    PetscScalar zsol_max_global[2];
	    START_PLANE_LOOP
        solidus[L][j][i] = zsol[L][j][i];
        if (zsol[L][j][i] > zsol_max_local[0]) {
            zsol_max_local[1] = zsol_max_local[0];
            zsol_max_local[0] = zsol[L][j][i];
        } else if (zsol[L][j][i] > zsol_max_local[1]) {
            zsol_max_local[1] = zsol[L][j][i];
        }
    END_PLANE_LOOP
	MPI_Allreduce(&zsol_max_local, &zsol_max_global, 2, MPIU_SCALAR, MPI_MAX, PETSC_COMM_WORLD); */

	// calculate depth average stresses, strain rate, and pressures
	START_PLANE_LOOP
	// first get melt pressure based on T structure at start of time step (during dike locate/phase transition)
	if (sFlag == 1) // *djking
	{
		magP = 0;									 // set magP to zero
/* 		magP = (-14) * (dike->drhomagma) * grav[2];	 // SET EXCESS MAGMA PRESSURE */
		magma_presence = 0;							 // testing
		if (dike->zmax_magma - solidus[L][j][i] < 0) // if negative, then postive magma pressure at solidus exists
		{
			magP = dike->magPMeltFrac*(dike->zmax_magma - solidus[L][j][i]) * (dike->drhomagma) * grav[2];									// excess magma pressure at solidus
			magma_presence = dike->magPfac * (solidus[L][j][i] - dike->zmax_magma) / (zsol_max_global - dike->zmax_magma); // undergoing testing
		}
		// gmagPressure[L][j][i] = (lithP[L][j][i]/liththick[L][j][i]+magP)*magma_presence;
		gmagPressure[L][j][i] = magP;						  // excess magma pressure
		magPresence[L][j][i] = magma_presence;				  // *testing
		focused_magPressure[L][j][i] = magP * magma_presence; // revisit and turn on magma_presence if needed *testing
	}
		// Brittle layer stresses calculated from T_brit (brittle-ductile transition)
		ghxx_ave[L][j][i] = hxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean history stress
		ghyy_ave[L][j][i] = hyy[L][j][i] / britthick[L][j][i]; // Depth weighted mean history stress
		gsxx_ave[L][j][i] = sxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean stress
		gsyy_ave[L][j][i] = syy[L][j][i] / britthick[L][j][i]; // Depth weighted mean stress
		gdxx_ave[L][j][i] = dxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean strain rate
		gdyy_ave[L][j][i] = dyy[L][j][i] / britthick[L][j][i]; // Depth weighted mean strain rate
		
		// *delete after debug
		gsxx_eff_ave[L][j][i] = ogsxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean total stress
		raw_gsxx[L][j][i] = ogsxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean total stress
		raw_gsxx_ave[L][j][i] = ogsxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean total stress
		smooth_gsxx[L][j][i] = ogsxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean total stress
		smooth_gsxx_ave[L][j][i] = ogsxx[L][j][i] / britthick[L][j][i]; // Depth weighted mean total stress

		// Lithospheric average pressure calculated from solidus
		ghP_ave[L][j][i] = hP[L][j][i] / liththick[L][j][i]; // Depth weighted mean history pressure
		gPc_ave[L][j][i] = Pc[L][j][i] / liththick[L][j][i]; // Depth weighted mean pressure
		glithP_ave[L][j][i] = lithP[L][j][i] / liththick[L][j][i]; // Depth weighted mean lithostatic pressure

		if (L == 0) // *djking *debugging
		{
			x_c = COORD_CELL(i, sx, fs->dsx);
			y_c = COORD_CELL(j, sy, fs->dsy);
			if (x_c < 0.3 && x_c > 0.0 && y_c == -1.5)
			{
				PetscCall(PetscPrintf(PETSC_COMM_WORLD, "liththick=%.4e, britthick=%.4e, zsol=%.4e, solidus=%.4e, horizontal stress=%.4e, magP=%.4e\n", liththick[L][j][i], britthick[L][j][i], zsol[L][j][i], solidus[L][j][i], gsxx_eff_ave[L][j][i], gmagPressure[L][j][i]));
			}
		}

	END_PLANE_LOOP


	// output lithospheric averaged stress arrays to .txt file on timesteps of other output
	if (((istep % nstep_out) == 0 || istep == 1) && (dike->out_stress > 0))
	{
		if (L == 0)
		{
			// Form the filename based on jr->ts->istep+1
			std::ostringstream oss;
			oss << "Compute_sxx_magP_outputs_Timestep_" << std::setfill('0') << std::setw(8) << (jr->ts->istep + 1) << ".txt";
			std::string filename = oss.str();

			// Open a file with the formed filename
			std::ofstream outFile(filename);
			if (outFile)
			{
				START_PLANE_LOOP
				x_c = COORD_CELL(i, sx, fs->dsx);
				y_c = COORD_CELL(j, sy, fs->dsy);

				// Writing space delimited data
				outFile
					<< " " << x_c << " " << y_c
					<< " " << ghxx_ave[L][j][i]
					<< " " << ghyy_ave[L][j][i]
					<< " " << gsxx_ave[L][j][i]
					<< " " << gsyy_ave[L][j][i]
					<< " " << gdxx_ave[L][j][i]
					<< " " << gdyy_ave[L][j][i]
					<< " " << britthick[L][j][i]
					<< " " << ghP_ave[L][j][i]
					<< " " << gPc_ave[L][j][i]
					<< " " << glithP_ave[L][j][i]
					<< " " << gmagPressure[L][j][i]
					<< " " << liththick[L][j][i]
					<< " " << gsxx_eff_ave[L][j][i] << "\n";

				END_PLANE_LOOP
			}
			else
			{
				std::cerr << "Error creating file: " << filename << std::endl;
			}
		}
	}
	
	// restore buffer and mean stress vectors
	ierr = DMDAVecRestoreArray(fs->DA_CEN, jr->lT,   &lT);  CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(fs->DA_CEN, jr->lp,   &lp);  CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(fs->DA_CEN, jr->lp_lith, &p_lith); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->solidus, &solidus); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->magPresence, &magPresence); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->focused_magPressure, &focused_magPressure); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->sxx_eff_ave, &gsxx_eff_ave); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->raw_sxx, &raw_gsxx); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->raw_sxx_ave, &raw_gsxx_ave); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->smooth_sxx, &smooth_gsxx); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->smooth_sxx_ave, &smooth_gsxx_ave); CHKERRQ(ierr);  // *delete after debug

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hxx_ave, &ghxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hyy_ave, &ghyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->sxx_ave, &gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->syy_ave, &gsyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->dxx_ave, &gdxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->dyy_ave, &gdyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hP_ave, &ghP_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->Pc_ave, &gPc_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->lithP_ave, &glithP_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->magPressure, &gmagPressure); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhxx, &hxx); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhyy, &hyy); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx, &sxx); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsyy, &syy); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdxx, &dxx); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdyy, &dyy); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhP, &hP); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vPc, &Pc); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vlithP, &lithP); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vliththick, &liththick); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vcdxx, &cdxx); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vcdyy, &cdyy); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vzsol, &zsol); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vogsxx, &ogsxx); CHKERRQ(ierr); // *delete after debug
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vbritthick, &britthick); CHKERRQ(ierr); // *djking

	ierr = VecRestoreArray(vhxx, &lhxx); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhyy, &lhyy); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx, &lsxx); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsyy, &lsyy); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdxx, &ldxx); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdyy, &ldyy); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhP, &lhP); CHKERRQ(ierr);
	ierr = VecRestoreArray(vPc, &lPc); CHKERRQ(ierr);
	ierr = VecRestoreArray(vlithP, &llithP); CHKERRQ(ierr);
	ierr = VecRestoreArray(vliththick, &lliththick); CHKERRQ(ierr);
	ierr = VecRestoreArray(vcdxx, &lcdxx); CHKERRQ(ierr);
	ierr = VecRestoreArray(vcdyy, &lcdyy); CHKERRQ(ierr);
	ierr = VecRestoreArray(vzsol, &lzsol); CHKERRQ(ierr);
	ierr = VecRestoreArray(vogsxx, &logsxx); CHKERRQ(ierr); // *delete after debug
	ierr = VecRestoreArray(vliththick, &lliththick); CHKERRQ(ierr); // *djking

	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhxx); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhyy); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsyy); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdxx); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdyy); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhP); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vPc); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vlithP); CHKERRQ(ierr);  
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vliththick); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vcdxx); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vcdyy); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vzsol); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vogsxx); CHKERRQ(ierr); // *delete after debug
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vbritthick); CHKERRQ(ierr); // *djking

	//fill ghost points

	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->magPresence);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->solidus);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->focused_magPressure);

	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->sxx_eff_ave); // *delete after debug
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->raw_sxx); // *delete after debug
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->raw_sxx_ave); // *delete after debug
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->smooth_sxx); // *delete after debug
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->smooth_sxx_ave); // *delete after debug

	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hxx_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hyy_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->sxx_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->syy_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->dxx_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->dyy_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hP_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->Pc_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->lithP_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->magPressure);

	PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------------
// Apply elliptical Gaussian smoothing of depth-averaged effective stress.  
// **NOTE** There is NO message passing between adjacent procs in x

PetscErrorCode Smooth_sxx_eff(JacRes *jr, PetscInt nD, PetscInt nPtr, PetscInt  j1, PetscInt j2, PetscInt sFlag)
{

	FDSTAG      *fs;
	Dike        *dike;
	Discret1D   *dsz, *dsy;
	FreeSurf    *surf;
	Ph_trans_t  *CurrPhTr;

	PetscScalar ***surface, ***solidus, lithick; // *djking
	PetscScalar ***gmagPressure, ***focused_magPressure, ***magPresence; // *djking

	PetscScalar ***ghxx_ave, ***ghxx_ave_hist, ***ghxx_ave_smooth; // *djking
	PetscScalar ***ghyy_ave, ***ghyy_ave_hist, ***ghyy_ave_smooth; // *djking
	PetscScalar ***gsxx_ave, ***gsxx_ave_hist, ***gsxx_ave_smooth; // *djking
	PetscScalar ***gsyy_ave, ***gsyy_ave_hist, ***gsyy_ave_smooth; // *djking
	PetscScalar ***gdxx_ave, ***gdxx_ave_hist, ***gdxx_ave_smooth; // *djking
	PetscScalar ***gdyy_ave, ***gdyy_ave_hist, ***gdyy_ave_smooth; // *djking
	PetscScalar ***ghP_ave, ***ghP_ave_hist, ***ghP_ave_smooth; // *djking
	PetscScalar ***gPc_ave, ***gPc_ave_hist, ***gPc_ave_smooth; // *djking
	PetscScalar ***glithP_ave, ***glithP_ave_hist, ***glithP_ave_smooth; // *djking
	PetscScalar ***gmagPressure_hist, ***gmagPressure_smooth; // *djking
	PetscScalar ***hxx_ave, *lhxx_ave, ***hxx_ave_prev, *lhxx_ave_prev, ***hxx_ave_next, *lhxx_ave_next; // *djking
	PetscScalar ***hyy_ave, *lhyy_ave, ***hyy_ave_prev, *lhyy_ave_prev, ***hyy_ave_next, *lhyy_ave_next; // *djking
	PetscScalar ***sxx_ave, *lsxx_ave, ***sxx_ave_prev, *lsxx_ave_prev, ***sxx_ave_next, *lsxx_ave_next; // *djking
	PetscScalar ***syy_ave, *lsyy_ave, ***syy_ave_prev, *lsyy_ave_prev, ***syy_ave_next, *lsyy_ave_next; // *djking
	PetscScalar ***dxx_ave, *ldxx_ave, ***dxx_ave_prev, *ldxx_ave_prev, ***dxx_ave_next, *ldxx_ave_next; // *djking
	PetscScalar ***dyy_ave, *ldyy_ave, ***dyy_ave_prev, *ldyy_ave_prev, ***dyy_ave_next, *ldyy_ave_next; // *djking
	PetscScalar ***hP_ave, *lhP_ave, ***hP_ave_prev, *lhP_ave_prev, ***hP_ave_next, *lhP_ave_next; // *djking
	PetscScalar ***Pc_ave, *lPc_ave, ***Pc_ave_prev, *lPc_ave_prev, ***Pc_ave_next, *lPc_ave_next; // *djking
	PetscScalar ***lithP_ave, *llithP_ave, ***lithP_ave_prev, *llithP_ave_prev, ***lithP_ave_next, *llithP_ave_next; // *djking
	PetscScalar sum_ave_hxx, sum_ave_hyy, sum_ave_sxx, sum_ave_syy, sum_ave_dxx, sum_ave_dyy; // *djking
	PetscScalar sum_ave_hP, sum_ave_Pc, sum_ave_lithP; // *djking
	Vec         vhxx_ave, vhxx_ave_prev, vhxx_ave_next; // *djking
	Vec         vhyy_ave, vhyy_ave_prev, vhyy_ave_next; // *djking
	Vec         vsxx_ave, vsxx_ave_prev, vsxx_ave_next; // *djking
	Vec         vsyy_ave, vsyy_ave_prev, vsyy_ave_next; // *djking
	Vec         vdxx_ave, vdxx_ave_prev, vdxx_ave_next; // *djking
	Vec         vdyy_ave, vdyy_ave_prev, vdyy_ave_next; // *djking
	Vec         vhP_ave, vhP_ave_prev, vhP_ave_next; // *djking
	Vec         vPc_ave, vPc_ave_prev, vPc_ave_next; // *djking
	Vec         vlithP_ave, vlithP_ave_prev, vlithP_ave_next; // *djking

	PetscScalar ***gsxx_eff_ave, ***gsxx_eff_ave_hist; // *delete after debug
	PetscScalar ***raw_gsxx, ***smooth_gsxx; // *delete after debug
	PetscScalar ***raw_gsxx_ave, ***raw_gsxx_ave_hist; // *delete after debug
	PetscScalar ***smooth_gsxx_ave, ***smooth_gsxx_ave_hist; // *delete after debug

	PetscScalar ***ycoors, *lycoors, ***ycoors_prev, *lycoors_prev, ***ycoors_next, *lycoors_next;
	PetscScalar ***xcenter, *lxcenter, ***xcenter_prev, *lxcenter_prev, ***xcenter_next, *lxcenter_next;
	PetscScalar ***sxx, *lsxx, ***sxx_prev, *lsxx_prev, ***sxx_next, *lsxx_next;
	PetscScalar ***magP, *lmagP, ***magP_prev, *lmagP_prev, ***magP_next, *lmagP_next;
	PetscScalar xc, yc, xx, yy, dx, dy, sum_sxx, sum_sxx_raw, sum_sxx_smooth, sum_magP, sum_w;
	PetscScalar filtx, filty, w, dfac, magPfac, magPwidth;
	PetscScalar xcent, xcent_north, xcent_south, ycent_north, ycent_south, xcent_search, ycent_search;
	PetscScalar azim, dalong, dxazim, dyazim, radbound, sumslope, sumadd;
	PetscScalar dx_tot, dy_tot, str_y;
	//PetscScalar dyazmin, dyazmax, dyaz;

	Vec         vycoors, vycoors_prev, vycoors_next;
	Vec         vxcenter, vxcenter_prev, vxcenter_next;
	Vec         vsxx, vsxx_prev, vsxx_next;
	Vec         vmagP, vmagP_prev, vmagP_next;

	PetscInt    j, jj, j1prev, j2prev, j1next, j2next, jj1, jj2; 
	PetscInt    i,ii, ii1, ii2;
	PetscInt    sx, sy, sz, nx, ny, nz;
	PetscInt    L, M;
	PetscMPIInt rank;
	PetscInt    sisc, istep_count, istep_nave, istep, nstep_out;

  
	PetscErrorCode ierr;
	PetscFunctionBeginUser;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Request srequest, rrequest, srequest2, rrequest2, srequest3, rrequest3, srequest4, rrequest4;
	MPI_Request srequest5, rrequest5, srequest6, rrequest6, srequest7, rrequest7, srequest8, rrequest8; // *djking
	MPI_Request srequest9, rrequest9, srequest10, rrequest10, srequest11, rrequest11, srequest12, rrequest12, srequest13, rrequest13; // *djking

	fs  =  jr->fs;
	dsz = &fs->dsz;
	dsy = &fs->dsy;
	L   =  (PetscInt)dsz->rank;
	M   =  (PetscInt)dsy->rank;

	istep=jr->ts->istep+1; 
	nstep_out=jr->ts->nstep_out;

	ierr = DMDAGetCorners(fs->DA_CEN, &sx, &sy, &sz, &nx, &ny, &nz); CHKERRQ(ierr);

	dike = jr->dbdike->matDike+nD;
	filtx=dike->filtx;
	filty=dike->filty;
	dfac=1.0; //maximum distance for Gaussian weights is dfac*filtx and dfac*filty

	magPfac=dike->magPfac;
	magPwidth=dike->magPwidth;
	CurrPhTr = jr->dbm->matPhtr+nPtr;

	surf = jr->surf;


// get communication buffer (Gets a PETSc vector, vycoors, that may be used with the DM global routines)
//y node coords
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vycoors); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vycoors_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vycoors_next); CHKERRQ(ierr);

	ierr = VecZeroEntries(vycoors); CHKERRQ(ierr);
	ierr = VecZeroEntries(vycoors_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vycoors_next); CHKERRQ(ierr);


//for dike center
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vxcenter); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vxcenter_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vxcenter_next); CHKERRQ(ierr);

	ierr = VecZeroEntries(vxcenter); CHKERRQ(ierr);
	ierr = VecZeroEntries(vxcenter_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vxcenter_next); CHKERRQ(ierr);


//sxx_ave info
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx_next); CHKERRQ(ierr);

	ierr = VecZeroEntries(vsxx); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsxx_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsxx_next); CHKERRQ(ierr); 

//magP info
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vmagP); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vmagP_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vmagP_next); CHKERRQ(ierr);

	ierr = VecZeroEntries(vmagP); CHKERRQ(ierr);
	ierr = VecZeroEntries(vmagP_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vmagP_next); CHKERRQ(ierr); 

// stress/strainrate/pressure info // *djking
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhxx_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhxx_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhxx_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhxx_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhxx_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhxx_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhyy_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhyy_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhyy_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhyy_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhyy_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhyy_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsxx_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsxx_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsxx_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsxx_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsyy_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsyy_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vsyy_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsyy_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsyy_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vsyy_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdxx_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdxx_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdxx_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vdxx_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vdxx_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vdxx_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdyy_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdyy_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vdyy_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vdyy_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vdyy_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vdyy_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhP_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhP_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vhP_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhP_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhP_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vhP_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vPc_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vPc_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vPc_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vPc_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vPc_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vPc_ave_next); CHKERRQ(ierr); 
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vlithP_ave); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vlithP_ave_prev); CHKERRQ(ierr);
	ierr = DMGetGlobalVector(jr->DA_CELL_2D, &vlithP_ave_next); CHKERRQ(ierr);
	ierr = VecZeroEntries(vlithP_ave); CHKERRQ(ierr);
	ierr = VecZeroEntries(vlithP_ave_prev); CHKERRQ(ierr);
	ierr = VecZeroEntries(vlithP_ave_next); CHKERRQ(ierr); 

// open index buffer for computation (ycoors is the array that shares data with vector vycoors & indexed with global dimensions)
//y node coords
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vycoors, &ycoors); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vycoors_prev, &ycoors_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vycoors_next, &ycoors_next); CHKERRQ(ierr);

//dike center info
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vxcenter, &xcenter); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vxcenter_prev, &xcenter_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vxcenter_next, &xcenter_next); CHKERRQ(ierr);

//sxx_ave info
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx, &sxx); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx_prev, &sxx_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx_next, &sxx_next); CHKERRQ(ierr);

//magP info
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vmagP, &magP); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vmagP_prev, &magP_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vmagP_next, &magP_next); CHKERRQ(ierr);

// stress/strainrate/pressure info // *djking
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhxx_ave, &hxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhxx_ave_prev, &hxx_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhxx_ave_next, &hxx_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhyy_ave, &hyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhyy_ave_prev, &hyy_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhyy_ave_next, &hyy_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx_ave, &sxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx_ave_prev, &sxx_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsxx_ave_next, &sxx_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsyy_ave, &syy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsyy_ave_prev, &syy_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vsyy_ave_next, &syy_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdxx_ave, &dxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdxx_ave_prev, &dxx_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdxx_ave_next, &dxx_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdyy_ave, &dyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdyy_ave_prev, &dyy_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vdyy_ave_next, &dyy_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhP_ave, &hP_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhP_ave_prev, &hP_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vhP_ave_next, &hP_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vPc_ave, &Pc_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vPc_ave_prev, &Pc_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vPc_ave_next, &Pc_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vlithP_ave, &lithP_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vlithP_ave_prev, &lithP_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, vlithP_ave_next, &lithP_ave_next); CHKERRQ(ierr);

// open linear buffer for send/receive  (returns the pointer, lsxx..., that contains this processor portion of vector data, vycoors)
//y node coords
	ierr = VecGetArray(vycoors, &lycoors); CHKERRQ(ierr);
	ierr = VecGetArray(vycoors_prev, &lycoors_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vycoors_next, &lycoors_next); CHKERRQ(ierr);
//celly_xbound info
	ierr = VecGetArray(vxcenter, &lxcenter); CHKERRQ(ierr);
	ierr = VecGetArray(vxcenter_prev, &lxcenter_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vxcenter_next, &lxcenter_next); CHKERRQ(ierr);

//sxx_ave info
	ierr = VecGetArray(vsxx, &lsxx); CHKERRQ(ierr);
	ierr = VecGetArray(vsxx_prev, &lsxx_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vsxx_next, &lsxx_next); CHKERRQ(ierr);
	
//magP info
	ierr = VecGetArray(vmagP, &lmagP); CHKERRQ(ierr);
	ierr = VecGetArray(vmagP_prev, &lmagP_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vmagP_next, &lmagP_next); CHKERRQ(ierr);
	
// stress/strainrate/pressure info // *djking
	ierr = VecGetArray(vhxx_ave, &lhxx_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vhxx_ave_prev, &lhxx_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vhxx_ave_next, &lhxx_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vhyy_ave, &lhyy_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vhyy_ave_prev, &lhyy_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vhyy_ave_next, &lhyy_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vsxx_ave, &lsxx_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vsxx_ave_prev, &lsxx_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vsxx_ave_next, &lsxx_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vsyy_ave, &lsyy_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vsyy_ave_prev, &lsyy_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vsyy_ave_next, &lsyy_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vdxx_ave, &ldxx_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vdxx_ave_prev, &ldxx_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vdxx_ave_next, &ldxx_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vdyy_ave, &ldyy_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vdyy_ave_prev, &ldyy_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vdyy_ave_next, &ldyy_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vhP_ave, &lhP_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vhP_ave_prev, &lhP_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vhP_ave_next, &lhP_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vPc_ave, &lPc_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vPc_ave_prev, &lPc_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vPc_ave_next, &lPc_ave_next); CHKERRQ(ierr);
	ierr = VecGetArray(vlithP_ave, &llithP_ave); CHKERRQ(ierr);
	ierr = VecGetArray(vlithP_ave_prev, &llithP_ave_prev); CHKERRQ(ierr);
	ierr = VecGetArray(vlithP_ave_next, &llithP_ave_next); CHKERRQ(ierr);

//access depth-averaged arrays on current proc
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->sxx_eff_ave, &gsxx_eff_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->raw_sxx, &raw_gsxx); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->raw_sxx_ave, &raw_gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->smooth_sxx, &smooth_gsxx); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->smooth_sxx_ave, &smooth_gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->magPressure, &gmagPressure); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->focused_magPressure, &focused_magPressure); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->solidus, &solidus); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->magPresence, &magPresence); CHKERRQ(ierr);	
	ierr = DMDAVecGetArray(surf->DA_SURF, surf->gtopo, &surface); CHKERRQ(ierr);

	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hxx_ave, &ghxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hyy_ave, &ghyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->sxx_ave, &gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->syy_ave, &gsyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->dxx_ave, &gdxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->dyy_ave, &gdyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hP_ave, &ghP_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->Pc_ave, &gPc_ave); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->lithP_ave, &glithP_ave); CHKERRQ(ierr);

	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hxx_ave_smooth, &ghxx_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hyy_ave_smooth, &ghyy_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->sxx_ave_smooth, &gsxx_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->syy_ave_smooth, &gsyy_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->dxx_ave_smooth, &gdxx_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->dyy_ave_smooth, &gdyy_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->hP_ave_smooth, &ghP_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->Pc_ave_smooth, &gPc_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->lithP_ave_smooth, &glithP_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->magPressure_smooth, &gmagPressure_smooth); CHKERRQ(ierr);
	
	START_PLANE_LOOP

		magP[L][j][i]=gmagPressure[L][j][i];
		sxx[L][j][i]=gsxx_eff_ave[L][j][i];
		
		hxx_ave[L][j][i]=ghxx_ave[L][j][i];
		hyy_ave[L][j][i]=ghyy_ave[L][j][i];
		sxx_ave[L][j][i]=gsxx_ave[L][j][i];
		syy_ave[L][j][i]=gsyy_ave[L][j][i];
		dxx_ave[L][j][i]=gdxx_ave[L][j][i];
		dyy_ave[L][j][i]=gdyy_ave[L][j][i];
		hP_ave[L][j][i]=ghP_ave[L][j][i];
		Pc_ave[L][j][i]=gPc_ave[L][j][i];
		lithP_ave[L][j][i]=glithP_ave[L][j][i];

/* 		if (L == 0) // *djking *debugging
		{
			xc = COORD_CELL(i, sx, fs->dsx);
			yc = COORD_CELL(j, sy, fs->dsy);
			if (xc < 0.3 && xc > 0.0 && yc == -1.5)
			{
				PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PRESMOOTH: gsxx=%.4e, sxx=%.4e, magP=%.4e\n", sxx[L][j][i] + magP[L][j][i], sxx[L][j][i], magP[L][j][i]));
			}
		} */

	END_PLANE_LOOP
  
//  Set up y-node coord and dike center arrays for passing between procs
	for(j = 0; j <= ny; j++)
	{
		xcenter[L][M][j]=1e+12;
		ycoors[L][M][j]=COORD_NODE(j+sy,sy,fs->dsy);  //can put j in last entry because ny<nx
	} 
//Dike center is given only on the current dike, i.e., j=j1 to j2
	for(j = j1; j <=j2; j++)
	{
		xcenter[L][M][j]=(CurrPhTr->celly_xboundR[j] + CurrPhTr->celly_xboundL[j])/2;    
	}

//--------------------------------------------------
// passing arrays between previous and next y proc
//--------------------------------------------------
	if (dsy->nproc > 1 && dsy->grprev != -1)  //Exchange arrays from previous proc, if not the first proc
	{
		ierr = MPI_Irecv(lycoors_prev, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lxcenter_prev, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest2); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest2, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Irecv(lsxx_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest3); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest3, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lmagP_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest4); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest4, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		// *djking
		ierr = MPI_Irecv(lhxx_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest5); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest5, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lhyy_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest6); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest6, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lsxx_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest7); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest7, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lsyy_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest8); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest8, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(ldxx_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest9); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest9, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(ldyy_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest10); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest10, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lhP_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest11); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest11, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lPc_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest12); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest12, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(llithP_ave_prev, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest13); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest13, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);


		ierr = MPI_Isend(lycoors, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lxcenter, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest2); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest2, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		
		ierr = MPI_Isend(lsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest3); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest3, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lmagP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest4); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest4, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		
		// *djking
		ierr = MPI_Isend(lhxx_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest5); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest5, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lhyy_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest6); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest6, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lsxx_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest7); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest7, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lsyy_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest8); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest8, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(ldxx_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest9); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest9, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(ldyy_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest10); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest10, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lhP_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest11); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest11, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lPc_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest12); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest12, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(llithP_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest13); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest13, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	}

	if ((dsy->nproc != 1) &&  (dsy->grnext != -1))  //Exhange arrays with next proc, if not the last proc
	{
		ierr = MPI_Isend(lycoors, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lxcenter, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest2); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest2, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Isend(lsxx, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest3); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest3, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lmagP, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest4); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest4, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		// *djking
		ierr = MPI_Isend(lhxx_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest5); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest5, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lhyy_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest6); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest6, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lsxx_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest7); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest7, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lsyy_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest8); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest8, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(ldxx_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest9); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest9, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(ldyy_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest10); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest10, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lhP_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest11); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest11, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(lPc_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest12); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest12, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Isend(llithP_ave, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest13); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest13, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Irecv(lycoors_next, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lxcenter_next, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest2); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest2, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Irecv(lsxx_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest3); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest3, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lmagP_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest4); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest4, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		// *djking
		ierr = MPI_Irecv(lhxx_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest5); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest5, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lhyy_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest6); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest6, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lsxx_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest7); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest7, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lsyy_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest8); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest8, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(ldxx_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest9); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest9, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(ldyy_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest10); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest10, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lhP_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest11); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest11, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(lPc_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest12); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest12, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
		ierr = MPI_Irecv(llithP_ave_next, (PetscMPIInt)(nx*ny), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest13); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest13, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	}

//--------------------------------------------------------------------------------------
// Gaussian filter with one elliptical axis oriented with local azimuth of dike zone
//--------------------------------------------------------------------------------------
	//loop over ybounds of current dike
	for(j = j1+sy; j <= j2+sy; j++) 
	{

		//Local azimuth of dike as the mean of all dike points within distance 
		//of filty of current dike point (xcent, yc)
		xcent=xcenter[L][M][j-sy];
		yc = COORD_CELL(j, sy, fs->dsy);
		sumslope=0;
		sumadd = 0; 
		//loop through full y domain to find all points of dike near (xcent,yc)
		for(jj = sy; jj < sy+ny; jj++)
		{
			//Current proc
			xcent_search=xcenter[L][M][jj-sy];  //beyond dike end this will be 1e12 so dalong>filty
			ycent_search=COORD_CELL(jj, sy, fs->dsy);
			dalong=sqrt(pow((xcent-xcent_search),2)+pow((yc-ycent_search),2));
			if (jj<j && dalong<=filty) 			//if south of current point
			{
				xcent_north=xcenter[L][M][jj-sy+1];
				ycent_north=COORD_CELL(jj+1, sy, fs->dsy);
				xcent_south=xcent_search;
				ycent_south=ycent_search;
				sumslope += (xcent_north-xcent_south)/(ycent_north-ycent_south);
				sumadd += 1;
			}
			else if (jj >j && dalong<=filty)  //if north of current point
			{
				xcent_north=xcent_search;
				ycent_north=ycent_search;
				xcent_south=xcenter[L][M][jj-sy-1];
				ycent_south=COORD_CELL(jj-1, sy, fs->dsy);
				sumslope += (xcent_north-xcent_south)/(ycent_north-ycent_south);
				sumadd += 1;
			}
			
			//NEXT proc
			if ( dsy->grnext != -1)
			{
				xcent_search=xcenter_next[L][M][jj-sy];  //if beyond dike end this will be 1e12 so dalong>filty
 				ycent_search=(ycoors_next[L][M][jj-sy+1]+ycoors_next[L][M][jj-sy])/2;
				dalong=sqrt(pow((xcent-xcent_search),2)+pow((yc-ycent_search),2));
				if (jj==sy && dalong<=filty) 			//if at southernmost cell of next proc
				{
					xcent_north=xcent_search;   
					ycent_north=ycent_search;
					xcent_south=xcenter[L][M][ny-1];  	//northernmost point of current proc (local index)
					ycent_south=COORD_CELL(ny+sy-1, sy, fs->dsy);  //uses global indexing
					sumslope += (xcent_north-xcent_south)/(ycent_north-ycent_south);
					sumadd += 1;
				}
				if (jj > sy && dalong<=filty)   		 //if north of southernmost cell of next proc
				{
					xcent_north=xcent_search;
					ycent_north=ycent_search;
					xcent_south=xcenter_next[L][M][jj-sy-1];
					ycent_south=(ycoors_next[L][M][jj-sy]+ycoors_next[L][M][jj-sy-1])/2;
					sumslope += (xcent_north-xcent_south)/(ycent_north-ycent_south);
					sumadd += 1;
				}
			}

			//Previous proc
			if ( dsy->grprev != -1)
			{
				xcent_search=xcenter_prev[L][M][jj-sy];  //if beyond dike end this will be 1e12 and dalong>filty
 				ycent_search=(ycoors_prev[L][M][jj-sy+1]+ycoors_prev[L][M][jj-sy])/2;
				dalong=sqrt(pow((xcent-xcent_search),2)+pow((yc-ycent_search),2));

				if (jj==sy+ny-1 && dalong<=filty) 		//if at northern most cell of prev proc
				{
					xcent_north=xcenter[L][M][0];   	//southernmost cell of current proc (local indexing)
					ycent_north=COORD_CELL(sy, sy, fs->dsy);  //uses global indexing
					xcent_south=xcent_search;  
					ycent_south=ycent_search;
					sumslope += (xcent_north-xcent_south)/(ycent_north-ycent_south);
					sumadd += 1;
				}
				if (jj < sy+ny-1 && dalong<=filty)    	//if south of northernmost cell of prev proc
				{
					xcent_north=xcenter_prev[L][M][jj-sy+1];  
					ycent_north=(ycoors_prev[L][M][jj-sy+2]+ycoors_prev[L][M][jj-sy+1])/2;
					xcent_south=xcent_search;  //northern most cell of prev proc
					ycent_south=ycent_search;
					sumslope += (xcent_north-xcent_south)/(ycent_north-ycent_south);
					sumadd += 1;
				}
			}
			
		} //done with loop over j to get mean azimuth
		azim=atan(sumslope/sumadd);

		//identify global y index of cells within dy_tot of yc on local and adjacent processors
		j1prev=ny+sy-1; j2prev=sy;
		j1next=ny+sy-1; j2next=sy;
		jj1=sy+ny-1; jj2=sy;
		//projected from slanted axis coords to get x & y grid distances needed to encompass dfac*filtx and dfac*filty
		dx_tot=(fabs(dfac*filtx*cos(azim))+fabs(dfac*filty*sin(azim)));
		dy_tot=(fabs(dfac*filtx*sin(azim))+fabs(dfac*filty*cos(azim)));  


		//dyazmin=1e6; dyazmax=-1e6;  //for detecting if near dike zone end
		//Loop over y to define area of Gaussian smoothing patch
		for(jj = sy; jj < sy+ny; jj++)
		{
			//Previous proc
 			yy=(ycoors_prev[L][M][jj-sy+1]+ycoors_prev[L][M][jj-sy])/2;
			if ( dsy->grprev != -1 && fabs(yc-yy) <= dy_tot && xcenter_prev[L][M][jj-sy] < 1.0e+12) 
			{
				j1prev=(PetscInt)min(j1prev,jj);   
				j2prev=(PetscInt)max(j2prev,jj);
			}
			/* dyaz=(yy-yc)/cos(azim);  //for stretching: if distance oriented with "azim" is within filty 
			if ( dsy->grprev != -1 && fabs(dyaz) <= filty && xcenter_prev[L][M][jj-sy] < 1.0e+12) 
			{
				dyazmin=(PetscScalar)min(dyaz,dyazmin);
			}*/

			//Next proc
			yy=(ycoors_next[L][M][jj-sy+1]+ycoors_next[L][M][jj-sy])/2;
			if (dsy->grnext != -1 && fabs(yy-yc) <= dy_tot && xcenter_next[L][M][jj-sy] < 1.0e+12)
			{
				j1next=(PetscInt)min(j1next,jj);   
				j2next=(PetscInt)max(j2next,jj);
			}
			/*dyaz=(yy-yc)/cos(azim);  //for stretching: if distance oriented with "azim" is within filty
			if (dsy->grnext != -1 && fabs(dyaz)<=filty && xcenter_next[L][M][jj-sy] < 1.0e+12)
			{
				dyazmax=(PetscScalar)max(dyaz,dyazmax);
			}*/

			//Current proc
			yy=COORD_CELL(jj, sy, fs->dsy);
			if (fabs(yy-yc) <= dy_tot && xcenter[L][M][jj-sy] < 1.0e+12)
			{
				jj1=(PetscInt)min(jj1,jj);
				jj2=(PetscInt)max(jj2,jj);
			}
			/*dyaz=(yy-yc)/cos(azim);  //for stretching: if distance oriented with "azim" is within filty 
			if (fabs(dyaz) <= filty && xcenter[L][M][jj-sy] < 1.0e+12)
			{
				dyazmin=(PetscScalar)min(dyaz,dyazmin);
				dyazmax=(PetscScalar)max(dyaz,dyazmax);
			}*/
		}  //end y loop for defining area of Gaussian smoothing patch

		str_y=1;
		//if ((dyazmax-dyazmin)<2*filty)       //if dike zone end limits the distance to < dfac*filty north or south
		//	str_y=2*filty/(dyazmax-dyazmin);  //then stretch filty smoothing extends a total distance 2*dfac*filty
		//str_y=(PetscScalar)min(str_y,2.0);
		/*if (L==0)  //debugging
		{ 
			PetscSynchronizedPrintf(PETSC_COMM_WORLD,"212121.2121 %lld M=%i: %i %g prev: %g to %g, curr: %g to %g next: %g to %g dytot=%g \n", 
			(LLD)(jr->ts->istep+1), M, j, yc, 
			(ycoors_prev[L][M][j1prev-sy+1]+ycoors_prev[L][M][j1prev-sy])/2, (ycoors_prev[L][M][j2prev-sy+1]+ycoors_prev[L][M][j2prev-sy])/2,
												COORD_CELL(jj1, sy, fs->dsy), COORD_CELL(jj2, sy, fs->dsy),
			(ycoors_next[L][M][j1next-sy+1]+ycoors_next[L][M][j1next-sy])/2,(ycoors_next[L][M][j2next-sy+1]+ycoors_next[L][M][j2next-sy])/2, dy_tot);

			//PetscSynchronizedPrintf(PETSC_COMM_WORLD,"2121.2121 %lld M=%i: j=%i, %g, %g \n", (LLD)(jr->ts->istep+1), M, j, ycoors_prev[L][M][j-sy], ycoors_prev[L][M][j-sy+1]); 
		}          //debugging
		*/

		//Loop over i to assign filtered value in cell j,i (again, one proc across all x dimension)
		for (i = sx; i < sx+nx; i++)  
		{
			sum_sxx=0.0;

			if (sFlag == 1) // magP is calculated from start of time step only
			{
				sum_magP = 0.0;
			}

			//*djking
			sum_ave_hxx=0.0;
			sum_ave_hyy=0.0;
			sum_ave_sxx=0.0;
			sum_ave_syy=0.0;
			sum_ave_dxx=0.0;
			sum_ave_dyy=0.0;
			sum_ave_hP=0.0;
			sum_ave_Pc=0.0;
			sum_ave_lithP=0.0;

			sum_w=0.0;

			xc =  COORD_CELL(i, sx, fs->dsx);
      
			//identify x cells within dfac*filtx of xc
			ii1=sx+nx-1; ii2=sx;
			for (ii = sx; ii < sx+nx; ii++)
			{
				xx = COORD_CELL(ii, sx, fs->dsx);

				if (fabs(xx-xc) <= dx_tot)
				{
					ii1=min(ii1,ii);
					ii2=max(ii2,ii);
				}
			}

			//weighted mean of values from previous proc
			for (jj = j1prev; jj <= j2prev; jj++) 
			{
				dy=ycoors_prev[L][M][jj+1-sy]-ycoors_prev[L][M][jj-sy];
				yy = (ycoors_prev[L][M][jj+1-sy] + ycoors_prev[L][M][jj-sy])/2;
				for (ii = ii1; ii <= ii2; ii++)
				{
					dx = SIZE_CELL(ii, sx, fs->dsx);
					xx = COORD_CELL(ii, sx, fs->dsx);

					dxazim=cos(azim)*(xx-xc)-sin(azim)*(yy-yc);
					dyazim=sin(azim)*(xx-xc)+cos(azim)*(yy-yc);

					radbound=(pow((dxazim/(dfac*filtx)),2) + pow((dyazim/(dfac*filty)),2));
					if (radbound<=1) //limit area of summing to within radbound of cell
					{
						w=exp(-0.5*(pow((dxazim/filtx),2) + pow((dyazim/(str_y*filty)),2)))*dx*dy;
						sum_sxx += sxx_prev[L][jj][ii]*w;

						if (sFlag == 1) // magP is calculated from start of time step only
						{
							sum_magP += magP_prev[L][jj][ii] * w;
						}

						// *djking
						sum_ave_hxx += hxx_ave_prev[L][jj][ii]*w;
						sum_ave_hyy += hyy_ave_prev[L][jj][ii]*w;
						sum_ave_sxx += sxx_ave_prev[L][jj][ii]*w;
						sum_ave_syy += syy_ave_prev[L][jj][ii]*w;
						sum_ave_dxx += dxx_ave_prev[L][jj][ii]*w;
						sum_ave_dyy += dyy_ave_prev[L][jj][ii]*w;
						sum_ave_hP += hP_ave_prev[L][jj][ii]*w;
						sum_ave_Pc += Pc_ave_prev[L][jj][ii]*w;
						sum_ave_lithP += lithP_ave_prev[L][jj][ii]*w;

						sum_w+=w;
					}
				}
			}//end loop over cells on previous proc

			//weighted mean of values on current proc
			for (jj = jj1; jj <= jj2; jj++)
			{
				dy=SIZE_CELL(jj,sy,fs->dsy);
				yy = COORD_CELL(jj,sy,fs->dsy);

				for (ii = ii1; ii <= ii2; ii++)
				{
					dx = SIZE_CELL(ii,sx,fs->dsx);
					xx = COORD_CELL(ii, sx, fs->dsx);

					dxazim=cos(azim)*(xx-xc)-sin(azim)*(yy-yc);
					dyazim=sin(azim)*(xx-xc)+cos(azim)*(yy-yc);

					radbound=(pow((dxazim/(dfac*filtx)),2) + pow((dyazim/(dfac*filty)),2));					
					if (radbound<=1)  //limit area of summing to within radbound of cell
					{
						w=exp(-0.5*(pow((dxazim/filtx),2) + pow((dyazim/(str_y*filty)),2)))*dx*dy;
						sum_sxx += sxx[L][jj][ii]*w;

						if (sFlag == 1) // magP is calculated from start of time step only
						{
							sum_magP += magP[L][jj][ii] * w;
						}

						// *djking
						sum_ave_hxx += hxx_ave[L][jj][ii]*w;
						sum_ave_hyy += hyy_ave[L][jj][ii]*w;
						sum_ave_sxx += sxx_ave[L][jj][ii]*w;
						sum_ave_syy += syy_ave[L][jj][ii]*w;
						sum_ave_dxx += dxx_ave[L][jj][ii]*w;
						sum_ave_dyy += dyy_ave[L][jj][ii]*w;
						sum_ave_hP += hP_ave[L][jj][ii]*w;
						sum_ave_Pc += Pc_ave[L][jj][ii]*w;
						sum_ave_lithP += lithP_ave[L][jj][ii]*w;

						sum_w+=w;
					}				
				}
			}//end loop over cells on current proc

			//weighted mean of values from next proc
			for (jj = j1next; jj <= j2next; jj++)
			{
				dy=ycoors_next[L][M][jj+1-sy]-ycoors_next[L][M][jj-sy];
				yy = (ycoors_next[L][M][jj+1-sy] + ycoors_next[L][M][jj-sy])/2;

				for (ii = ii1; ii <= ii2; ii++)
				{
					dx = SIZE_CELL(ii,sx,fs->dsx);
					xx = COORD_CELL(ii, sx, fs->dsx);

					dxazim=cos(azim)*(xx-xc)-sin(azim)*(yy-yc);
					dyazim=sin(azim)*(xx-xc)+cos(azim)*(yy-yc);

					radbound=(pow((dxazim/(dfac*filtx)),2) + pow((dyazim/(dfac*filty)),2));					
					if (radbound<=1)  //limit area of summing to within radbound of cell
					{
						w=exp(-0.5*(pow((dxazim/filtx),2) + pow((dyazim/(str_y*filty)),2)))*dx*dy;
						sum_sxx += sxx_next[L][jj][ii]*w;

						if (sFlag == 1) // magP is calculated from start of time step only
						{
							sum_magP += magP_next[L][jj][ii] * w;
						}

						// *djking
						sum_ave_hxx += hxx_ave_next[L][jj][ii]*w;
						sum_ave_hyy += hyy_ave_next[L][jj][ii]*w;
						sum_ave_sxx += sxx_ave_next[L][jj][ii]*w;
						sum_ave_syy += syy_ave_next[L][jj][ii]*w;
						sum_ave_dxx += dxx_ave_next[L][jj][ii]*w;
						sum_ave_dyy += dyy_ave_next[L][jj][ii]*w;
						sum_ave_hP += hP_ave_next[L][jj][ii]*w;
						sum_ave_Pc += Pc_ave_next[L][jj][ii]*w;
						sum_ave_lithP += lithP_ave_next[L][jj][ii]*w;

						sum_w+=w;
					}
				}
			} //end loop over cells from next proc

			//sum_w=max(sum_w,0.0);  //why would sum_w be <0???!
			focused_magPressure[L][j][i]=(sum_magP/sum_w)*magPfac*exp(-0.5*(pow((cos(azim)*(xcent-xc)/magPwidth),2)));
			smooth_gsxx[L][j][i]=(sum_sxx/sum_w);
			smooth_gsxx_ave[L][j][i]=(sum_sxx/sum_w);

			if (sFlag == 1) // magP is calculated from start of time step only
			{
				gmagPressure[L][j][i] = (sum_magP / sum_w);
			}

			gsxx_eff_ave[L][j][i]=(sum_sxx/sum_w) + gmagPressure[L][j][i];
			//gsxx_eff_ave[L][j][i]=(sum_sxx/sum_w) + focused_magPressure[L][j][i]; // *testing
			
			// *djking
			ghxx_ave_smooth[L][j][i]=(sum_ave_hxx/sum_w);
			ghyy_ave_smooth[L][j][i]=(sum_ave_hyy/sum_w);
			gsxx_ave_smooth[L][j][i]=(sum_ave_sxx/sum_w);
			gsyy_ave_smooth[L][j][i]=(sum_ave_syy/sum_w);
			gdxx_ave_smooth[L][j][i]=(sum_ave_dxx/sum_w);
			gdyy_ave_smooth[L][j][i]=(sum_ave_dyy/sum_w);
			ghP_ave_smooth[L][j][i]=(sum_ave_hP/sum_w);
			gPc_ave_smooth[L][j][i]=(sum_ave_Pc/sum_w);
			glithP_ave_smooth[L][j][i]=(sum_ave_lithP/sum_w);

			if (sFlag == 1) // magP is calculated from start of time step only
			{
				gmagPressure_smooth[L][j][i] = (sum_magP / sum_w);
			}

/* 			if (L == 0) // *djking *debugging
			{
				xc = COORD_CELL(i, sx, fs->dsx);
				yc = COORD_CELL(j, sy, fs->dsy);
				if (xc < 0.3 && xc > 0.0 && yc == -1.5)
				{
					PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PRETIMEAVESMOOTH: gsxx=%.4e, sxx=%.4e, magP=%.4e\n", gsxx_eff_ave[L][j][i], gsxx_ave_smooth[L][j][i], gmagPressure_smooth[L][j][i]));
				}
			} */

		}//End loop over i
	}// End loop over j

  	//PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT); // debugging All procs must run this

	//restore ycoors arrays
	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vycoors_prev, &ycoors_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vycoors_prev, &lycoors_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vycoors_prev); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vycoors, &ycoors); CHKERRQ(ierr);
	ierr = VecRestoreArray(vycoors, &lycoors); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vycoors); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vycoors_next, &ycoors_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vycoors_next, &lycoors_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vycoors_next); CHKERRQ(ierr);

	//restore xcenter arrays
	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vxcenter_prev, &xcenter_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vxcenter_prev, &lxcenter_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vxcenter_prev); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vxcenter, &xcenter); CHKERRQ(ierr);
	ierr = VecRestoreArray(vxcenter, &lxcenter); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vxcenter); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vxcenter_next, &xcenter_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vxcenter_next, &lxcenter_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vxcenter_next); CHKERRQ(ierr);

	//restore magP arrays
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vmagP_prev, &magP_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vmagP_prev, &lmagP_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vmagP_prev); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vmagP, &magP); CHKERRQ(ierr);
	ierr = VecRestoreArray(vmagP, &lmagP); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vmagP); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vmagP_next, &magP_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vmagP_next, &lmagP_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vmagP_next); CHKERRQ(ierr);

	//restore sxx arrays
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx_prev, &sxx_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx_prev, &lsxx_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx_prev); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx, &sxx); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx, &lsxx); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx_next, &sxx_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx_next, &lsxx_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx_next); CHKERRQ(ierr);

	//restore stress/strainrate/pressure arrays
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhxx_ave_prev, &hxx_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhxx_ave_prev, &lhxx_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhxx_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhxx_ave, &hxx_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhxx_ave, &lhxx_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhxx_ave_next, &hxx_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhxx_ave_next, &lhxx_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhxx_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhyy_ave_prev, &hyy_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhyy_ave_prev, &lhyy_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhyy_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhyy_ave, &hyy_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhyy_ave, &lhyy_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhyy_ave_next, &hyy_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhyy_ave_next, &lhyy_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhyy_ave_next); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx_ave_prev, &sxx_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx_ave_prev, &lsxx_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx_ave, &sxx_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx_ave, &lsxx_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsxx_ave_next, &sxx_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsxx_ave_next, &lsxx_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsxx_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsyy_ave_prev, &syy_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsyy_ave_prev, &lsyy_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsyy_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsyy_ave, &syy_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsyy_ave, &lsyy_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vsyy_ave_next, &syy_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vsyy_ave_next, &lsyy_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vsyy_ave_next); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdxx_ave_prev, &dxx_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdxx_ave_prev, &ldxx_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdxx_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdxx_ave, &dxx_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdxx_ave, &ldxx_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdxx_ave_next, &dxx_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdxx_ave_next, &ldxx_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdxx_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdyy_ave_prev, &dyy_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdyy_ave_prev, &ldyy_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdyy_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdyy_ave, &dyy_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdyy_ave, &ldyy_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vdyy_ave_next, &dyy_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vdyy_ave_next, &ldyy_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vdyy_ave_next); CHKERRQ(ierr);
	
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhP_ave_prev, &hP_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhP_ave_prev, &lhP_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhP_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhP_ave, &hP_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhP_ave, &lhP_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhP_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vhP_ave_next, &hP_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vhP_ave_next, &lhP_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vhP_ave_next); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vPc_ave_prev, &Pc_ave_prev); CHKERRQ(ierr);
	ierr = VecRestoreArray(vPc_ave_prev, &lPc_ave_prev); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vPc_ave_prev); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vPc_ave, &Pc_ave); CHKERRQ(ierr);
	ierr = VecRestoreArray(vPc_ave, &lPc_ave); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vPc_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, vPc_ave_next, &Pc_ave_next); CHKERRQ(ierr);
	ierr = VecRestoreArray(vPc_ave_next, &lPc_ave_next); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_2D, &vPc_ave_next); CHKERRQ(ierr);

//--------------------------------------------------
//  TIME Averaging
//--------------------------------------------------
	if (dike->istep_nave > 1)
	{
		ierr = DMDAGetCorners(jr->DA_CELL_2D_tave, &sx, &sy, &sisc, &nx, &ny, &istep_nave); CHKERRQ(ierr);

		if(istep_nave!=dike->istep_nave) 
		{
			SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Problems: istep_nave=%lld, dike->istep_nave=%lld\n", 
			(LLD)(istep_nave), (LLD)(dike->istep_nave));
		}

		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->sxx_eff_ave_hist, &gsxx_eff_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->raw_sxx_ave_hist, &raw_gsxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->smooth_sxx_ave_hist, &smooth_gsxx_ave_hist); CHKERRQ(ierr);
		
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->hxx_ave_hist, &ghxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->hyy_ave_hist, &ghyy_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->sxx_ave_hist, &gsxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->syy_ave_hist, &gsyy_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->dxx_ave_hist, &gdxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->dyy_ave_hist, &gdyy_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->hP_ave_hist, &ghP_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->Pc_ave_hist, &gPc_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->lithP_ave_hist, &glithP_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecGetArray(jr->DA_CELL_2D_tave, dike->magPressure_hist, &gmagPressure_hist); CHKERRQ(ierr);

		dike->istep_count++;

		if (dike->istep_count+1 > istep_nave) 
		{
			dike->istep_count=0;
		}  
    
		for (j = j1+sy; j <= j2+sy; j++ )  //Global coordinates
		{ 
			for (i=sx; i < sx+ny; i++)
			{
				sum_sxx=0;
				sum_sxx_raw = 0.0;
				sum_sxx_smooth = 0.0;
				gsxx_eff_ave_hist[sisc+dike->istep_count][j][i]=gsxx_eff_ave[L][j][i];  //array for current step
				raw_gsxx_ave_hist[sisc+dike->istep_count][j][i]=raw_gsxx_ave[L][j][i];  //array for current step
				smooth_gsxx_ave_hist[sisc+dike->istep_count][j][i]=smooth_gsxx_ave[L][j][i];  //array for current step
				
				//*djking
				sum_ave_hxx = 0.0;
				sum_ave_hyy = 0.0;
				sum_ave_sxx = 0.0;
				sum_ave_syy = 0.0;
				sum_ave_dxx = 0.0;
				sum_ave_dyy = 0.0;
				sum_ave_hP = 0.0;
				sum_ave_Pc = 0.0;
				sum_ave_lithP = 0.0;
/* 				if (sFlag == 1) // magP is calculated from start of time step only
				{ */
					sum_magP = 0.0;
/* 				} */
				ghxx_ave_hist[sisc+dike->istep_count][j][i]=ghxx_ave_smooth[L][j][i];  //array for current step
				ghyy_ave_hist[sisc+dike->istep_count][j][i]=ghyy_ave_smooth[L][j][i];  //array for current step
				gsxx_ave_hist[sisc+dike->istep_count][j][i]=gsxx_ave_smooth[L][j][i];  //array for current step
				gsyy_ave_hist[sisc+dike->istep_count][j][i]=gsyy_ave_smooth[L][j][i];  //array for current step
				gdxx_ave_hist[sisc+dike->istep_count][j][i]=gdxx_ave_smooth[L][j][i];  //array for current step
				gdyy_ave_hist[sisc+dike->istep_count][j][i]=gdyy_ave_smooth[L][j][i];  //array for current step
				ghP_ave_hist[sisc+dike->istep_count][j][i]=ghP_ave_smooth[L][j][i];  //array for current step
				gPc_ave_hist[sisc+dike->istep_count][j][i]=gPc_ave_smooth[L][j][i];  //array for current step
				glithP_ave_hist[sisc+dike->istep_count][j][i]=glithP_ave_smooth[L][j][i];  //array for current step
				gmagPressure_hist[sisc+dike->istep_count][j][i]=gmagPressure_smooth[L][j][i];  //array for current step
				
				for (istep_count=sisc; istep_count<sisc+istep_nave; istep_count++)
				{
 					sum_sxx+=gsxx_eff_ave_hist[istep_count][j][i];
 					sum_sxx_raw+=raw_gsxx_ave_hist[istep_count][j][i];
 					sum_sxx_smooth+=smooth_gsxx_ave_hist[istep_count][j][i];
					 
					 // *djking
 					sum_ave_hxx+=ghxx_ave_hist[istep_count][j][i];
 					sum_ave_hyy+=ghyy_ave_hist[istep_count][j][i];
 					sum_ave_sxx+=gsxx_ave_hist[istep_count][j][i];
 					sum_ave_syy+=gsyy_ave_hist[istep_count][j][i];
 					sum_ave_dxx+=gdxx_ave_hist[istep_count][j][i];
 					sum_ave_dyy+=gdyy_ave_hist[istep_count][j][i];
 					sum_ave_hP+=ghP_ave_hist[istep_count][j][i];
 					sum_ave_Pc+=gPc_ave_hist[istep_count][j][i];
 					sum_ave_lithP+=glithP_ave_hist[istep_count][j][i];

					if (sFlag == 1) // magP is calculated from start of time step only
					{
						sum_magP += gmagPressure_hist[istep_count][j][i];
					}
				}

				gsxx_eff_ave[L][j][i]=sum_sxx/((PetscScalar)istep_nave);
				raw_gsxx_ave[L][j][i]=sum_sxx_raw/((PetscScalar)istep_nave);
				smooth_gsxx_ave[L][j][i]=sum_sxx_smooth/((PetscScalar)istep_nave);
				
				// *djking
				ghxx_ave_smooth[L][j][i]=sum_ave_hxx/((PetscScalar)istep_nave);
				ghyy_ave_smooth[L][j][i]=sum_ave_hyy/((PetscScalar)istep_nave);
				gsxx_ave_smooth[L][j][i]=sum_ave_sxx/((PetscScalar)istep_nave);
				gsyy_ave_smooth[L][j][i]=sum_ave_syy/((PetscScalar)istep_nave);
				gdxx_ave_smooth[L][j][i]=sum_ave_dxx/((PetscScalar)istep_nave);
				gdyy_ave_smooth[L][j][i]=sum_ave_dyy/((PetscScalar)istep_nave);
				ghP_ave_smooth[L][j][i]=sum_ave_hP/((PetscScalar)istep_nave);
				gPc_ave_smooth[L][j][i]=sum_ave_Pc/((PetscScalar)istep_nave);
				glithP_ave_smooth[L][j][i]=sum_ave_lithP/((PetscScalar)istep_nave);

				if (sFlag == 1) // magP is calculated from start of time step only
				{
					gmagPressure_smooth[L][j][i] = sum_magP / ((PetscScalar)istep_nave);
				}

/* 				if (L == 0) // *djking *debugging
				{
					xc = COORD_CELL(i, sx, fs->dsx);
					yc = COORD_CELL(j, sy, fs->dsy);
					if (xc < 0.3 && xc > 0.0 && yc == -1.5)
					{
						PetscCall(PetscPrintf(PETSC_COMM_WORLD, "POSTTIMEAVESMOOTH: gsxx=%.4e, sxx=%.4e?, magP=%.4e?\n", gsxx_eff_ave[L][j][i], ghxx_ave_smooth[L][j][i]+ghP_ave_smooth[L][j][i], gmagPressure_smooth[L][j][i]));
					}
				} */
			}
		}

		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->sxx_eff_ave_hist, &gsxx_eff_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->raw_sxx_ave_hist, &raw_gsxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->smooth_sxx_ave_hist, &smooth_gsxx_ave_hist); CHKERRQ(ierr);
		
		//*djking
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->hxx_ave_hist, &ghxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->hyy_ave_hist, &ghyy_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->sxx_ave_hist, &gsxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->syy_ave_hist, &gsyy_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->dxx_ave_hist, &gdxx_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->dyy_ave_hist, &gdyy_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->hP_ave_hist, &ghP_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->Pc_ave_hist, &gPc_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->lithP_ave_hist, &glithP_ave_hist); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(jr->DA_CELL_2D_tave, dike->magPressure_hist, &gmagPressure_hist); CHKERRQ(ierr);
	}// end if nstep_ave>1

  // output smoothed stress array to .txt file on timesteps of other output
  if (((istep % nstep_out) == 0 || istep == 1) && (dike->out_stress > 0)) 
  {
    if (L == 0)
    {
      // Form the filename based on jr->ts->istep+1
      std::ostringstream oss;
      oss << "Smooth_sxx_eff_outputs_Timestep_" << std::setfill('0') << std::setw(8) << (jr->ts->istep+1) << ".txt";
      std::string filename = oss.str();

      // Open a file with the formed filename
      std::ofstream outFile(filename);
      if (outFile)
      {
        START_PLANE_LOOP
        xc = COORD_CELL(i, sx, fs->dsx);
        yc = COORD_CELL(j, sy, fs->dsy);
		lithick = surface[L][j][i] - solidus[L][j][i];

        // Writing space delimited data
        outFile
          << " " << xc << " " << yc 
          << " " << ghxx_ave_smooth[L][j][i]
          << " " << ghyy_ave_smooth[L][j][i]
          << " " << gsxx_ave_smooth[L][j][i]
          << " " << gsyy_ave_smooth[L][j][i]
          << " " << gdxx_ave_smooth[L][j][i]
          << " " << gdyy_ave_smooth[L][j][i]
          << " " << ghP_ave_smooth[L][j][i]
          << " " << gPc_ave_smooth[L][j][i]
          << " " << glithP_ave_smooth[L][j][i]
          << " " << gmagPressure_smooth[L][j][i] 
          << " " << gsxx_eff_ave[L][j][i] 
          << " " << jr->ts->istep+1 << " " << jr->ts->time * jr->scal->time    
		  << " " << surface[L][j][i] << " " << solidus[L][j][i]    
		  << " " << lithick << "\n";    
/*           << " " << xc << " " << yc 
          << " " << gmagPressure[L][j][i] 
          << " " << focused_magPressure[L][j][i]
          << " " << raw_gsxx[L][j][i] 
          << " " << raw_gsxx_ave[L][j][i] 
          << " " << smooth_gsxx[L][j][i] 
          << " " << smooth_gsxx_ave[L][j][i] 
          << " " << gsxx_eff_ave[L][j][i] 
          << " " << jr->ts->istep+1 << " " << jr->ts->time * jr->scal->time    
		  << " " << surface[L][j][i] << " " << solidus[L][j][i]    
		  << " " << lithick << " " << magPresence[L][j][i] << "\n";    */ 

        END_PLANE_LOOP
      }
      else
      {
        std::cerr << "Error creating file: " << filename << std::endl;
      }
    }
  }  

	//restore arrays
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->focused_magPressure, &focused_magPressure); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->sxx_eff_ave, &gsxx_eff_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->raw_sxx, &raw_gsxx); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->raw_sxx_ave, &raw_gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->smooth_sxx, &smooth_gsxx); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->smooth_sxx_ave, &smooth_gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->solidus, &solidus); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->magPresence, &magPresence); CHKERRQ(ierr); 
	ierr = DMDAVecRestoreArray(surf->DA_SURF, surf->gtopo, &surface); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->magPressure, &gmagPressure); CHKERRQ(ierr);
	
	// *djking
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hxx_ave, &ghxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hyy_ave, &ghyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->sxx_ave, &gsxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->syy_ave, &gsyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->dxx_ave, &gdxx_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->dyy_ave, &gdyy_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hP_ave, &ghP_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->Pc_ave, &gPc_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->lithP_ave, &glithP_ave); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hxx_ave_smooth, &ghxx_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hyy_ave_smooth, &ghyy_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->sxx_ave_smooth, &gsxx_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->syy_ave_smooth, &gsyy_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->dxx_ave_smooth, &gdxx_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->dyy_ave_smooth, &gdyy_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->hP_ave_smooth, &ghP_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->Pc_ave_smooth, &gPc_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->lithP_ave_smooth, &glithP_ave_smooth); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->magPressure_smooth, &gmagPressure_smooth); CHKERRQ(ierr);
	
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->sxx_eff_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->solidus);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->magPresence);
	
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hxx_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hyy_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hxx_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hyy_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->sxx_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->syy_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->sxx_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->syy_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->dxx_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->dyy_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->dxx_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->dyy_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hP_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->Pc_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->lithP_ave);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->magPressure);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->hP_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->Pc_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->lithP_ave_smooth);
	LOCAL_TO_LOCAL(jr->DA_CELL_2D, dike->magPressure_smooth);

	PetscFunctionReturn(0);  
}  

//----------------------------------------------------------------------------------------------------
// Set bounds of NotInAir box based on peak sxx_eff_ave
// NOTE that NOW, this only works if cpu_x =1
//

PetscErrorCode Set_dike_zones(JacRes *jr, PetscInt nD, PetscInt nPtr, PetscInt j1, PetscInt j2)
{

	FDSTAG      *fs;
	Dike        *dike;
	Discret1D   *dsx, *dsy, *dsz;
	Ph_trans_t  *CurrPhTr;
	PetscScalar ***gsxx_eff_ave;
	PetscScalar xcenter, sxx_max, dike_width, mindist, xshift, xcell;
	PetscScalar ***xboundL_pass, *lxboundL_pass, ***xboundR_pass, *lxboundR_pass;
	Vec         vxboundL_pass, vxboundR_pass;
	PetscInt    i, lj, j, sx, sy, sz, nx, ny, nz, L, Lx, M, ixcenter;
	PetscScalar sxxm, sxxp, dx12, dsdx1, dsdx2, x_maxsxx, ycell, dtime;   
	PetscInt    ixmax, istep, nstep_out;
 	MPI_Request srequest, rrequest;

	PetscErrorCode ierr;
	PetscFunctionBeginUser;

	fs  =  jr->fs;
	dsz = &fs->dsz;
	dsy = &fs->dsy;
	dsx = &fs->dsx;
	L   =  (PetscInt)dsz->rank;
	M   =  (PetscInt)dsy->rank;
	Lx  =  (PetscInt)dsx->rank;

	istep=jr->ts->istep+1; 
	nstep_out=jr->ts->nstep_out;

	dike = jr->dbdike->matDike+nD;
	CurrPhTr = jr->dbm->matPhtr+nPtr;
	dtime=jr->scal->time*jr->ts->time;


	if (Lx>0)
	{
		PetscPrintf(PETSC_COMM_WORLD,"Set_dike_zones requires cpu_x = 1 Lx = %lld \n", (LLD)(Lx));
		SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Set_dike_zones requires cpu_x = 1 Lx = %lld \n", (LLD)(Lx));
	}
	ierr = DMDAVecGetArray(jr->DA_CELL_2D, dike->sxx_eff_ave, &gsxx_eff_ave); CHKERRQ(ierr);
	ierr = DMDAGetCorners(fs->DA_CEN, &sx, &sy, &sz, &nx, &ny, &nz); CHKERRQ(ierr);
                                        
	for(lj = j1; lj <= j2; lj++)  //local index
	{
		sxx_max=-1e12;
		mindist=1e12;
		ixcenter = 0;
		xshift=0;
		ixmax=sx+1;

		j=sy+lj;  //global index
		dike_width=CurrPhTr->celly_xboundR[lj]-CurrPhTr->celly_xboundL[lj];
		xcenter=(CurrPhTr->celly_xboundR[lj] + CurrPhTr->celly_xboundL[lj])/2;

		for(i=sx+1; i < sx+nx-1; i++) 
		{
			xcell=COORD_CELL(i, sx, fs->dsx);
			if (fabs(xcell-xcenter) <= mindist) //find indice of dike zone center (xcenter)
			{
				ixcenter=i;
				mindist=fabs(xcell-xcenter);
			}
		} //end loop to find ixcenter
 
		for(i=ixcenter-2; i <= ixcenter+2; i++) //find max gsxx_eff at each value of y
		{
			if ((gsxx_eff_ave[L][j][i] > sxx_max))
			{
				sxx_max=gsxx_eff_ave[L][j][i];
				//xshift=COORD_CELL(i, sx, fs->dsx)-xcenter;
				ixmax=i;
			}
   		} 
		
		//finding where slope of dsxx/dx=0
		sxxm =  gsxx_eff_ave[L][j][ixmax-1];  //left of maximum point
		sxxp =  gsxx_eff_ave[L][j][ixmax+1]; ;  //right of max. point
 
		dsdx1=(sxx_max-sxxm)/(COORD_CELL(ixmax, sx, fs->dsx)-COORD_CELL(ixmax-1, sx, fs->dsx));  //slope left of max
		dsdx2=(sxxp-sxx_max)/(COORD_CELL(ixmax+1, sx, fs->dsx)-COORD_CELL(ixmax, sx, fs->dsx));  //slope right of max
		dx12=(COORD_CELL(ixmax+1, sx, fs->dsx)-COORD_CELL(ixmax-1, sx, fs->dsx))/2;

		if ((dsdx1>0) & (dsdx2<0))  //if local maximum, interpolate to find where dsdx=0;
		{
        	x_maxsxx=(COORD_CELL(ixmax-1, sx, fs->dsx)+COORD_CELL(ixmax, sx, fs->dsx))/2-dsdx1/(dsdx2-dsdx1)*dx12;
		}
		else  //just higher on either side of dike
		{
        	x_maxsxx=COORD_CELL(ixmax,sx,fs->dsx);
		}

		xshift=x_maxsxx-xcenter;

		if (xshift>0 && fabs(xshift) > 0.5*SIZE_CELL(ixcenter, sx, fs->dsx)) //ensure new center is within width of cell to right of center
		{
        	xshift=0.5*SIZE_CELL(ixcenter, sx, fs->dsx);
		}
		else if (xshift<0 && fabs(xshift) > 0.5*SIZE_CELL(ixcenter-1, sx, fs->dsx)) //ensure its within the width of cell left of center
		{
        	xshift=-0.5*SIZE_CELL(ixcenter-1, sx, fs->dsx);
		}

		//relocating dike bounds here
		CurrPhTr->celly_xboundL[lj]=xcenter+xshift-dike_width/2; 
		CurrPhTr->celly_xboundR[lj]=xcenter+xshift+dike_width/2; 

 // dike location to .txt file on timesteps of other output
       if (L==0 &&  ((istep % nstep_out) == 0 || istep == 1) && (dike->out_dikeloc > 0))
    {
      // Form the filename based on jr->ts->istep
      std::ostringstream oss;
      oss << "dikeloc_Timestep_" << (jr->ts->istep+1) << ".txt";

      std::string filename = oss.str();

      // Create/open file
      std::ofstream outFile(filename);
      if (outFile)
      {
        ycell = COORD_CELL(j, sy, fs->dsy);
        xcell=(COORD_CELL(ixmax-1, sx, fs->dsx)+COORD_CELL(ixmax, sx, fs->dsx))/2;

        // Writing space delimited data
        outFile
          << xcell << " " << ycell 
          << " " << xcenter << " " << xshift << " " << x_maxsxx 
          << " " << COORD_CELL(ixmax, sx, fs->dsx) 
          << " " << CurrPhTr->celly_xboundL[lj] 
          << " " << CurrPhTr->celly_xboundR[lj] 
          << " " << nD << " " << dtime << "\n"; 
      }
      else
      {
        std::cerr << "Error creating file: " << filename << std::endl;
      }
    }
  } //end loop over j cell row

	if (((istep % nstep_out)==0 || istep == 1) && (dike->out_dikeloc > 0))  
	{
		PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);
	}
	ierr = DMDAVecRestoreArray(jr->DA_CELL_2D, dike->sxx_eff_ave, &gsxx_eff_ave); CHKERRQ(ierr);

//-----------------------------------------------------------------------------------
// Set locations of ghost nodes
//-----------------------------------------------------------------------------------
	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vxboundL_pass); CHKERRQ(ierr);
	ierr = VecZeroEntries(vxboundL_pass); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vxboundL_pass, &xboundL_pass); CHKERRQ(ierr);
	ierr = VecGetArray(vxboundL_pass, &lxboundL_pass); CHKERRQ(ierr);

	ierr = DMGetGlobalVector(jr->DA_CELL_1D, &vxboundR_pass); CHKERRQ(ierr);
	ierr = VecZeroEntries(vxboundR_pass); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(jr->DA_CELL_1D, vxboundR_pass, &xboundR_pass); CHKERRQ(ierr);
	ierr = VecGetArray(vxboundR_pass, &lxboundR_pass); CHKERRQ(ierr);

      //Northernmost (top) ghost coord of northernmost (top) proc
	if (dsy->grnext == -1)
      {
		CurrPhTr->celly_xboundL[ny] = CurrPhTr->celly_xboundL[ny-1];
		CurrPhTr->celly_xboundR[ny] = CurrPhTr->celly_xboundR[ny-1];
	}
	//Southernmost ghost coord of southernmost (bottom) proc
	if (dsy->grprev == -1)  
	{
		CurrPhTr->celly_xboundL[-1] = CurrPhTr->celly_xboundL[0];
		CurrPhTr->celly_xboundR[-1] = CurrPhTr->celly_xboundR[0];
	}

	//Receive from 2nd northernmost (top in y) proc to southernmost (bottom in y) and set bottom ghost node
	if (dsy->nproc > 1 && dsy->grnext != -1) //MPI_Wait will make this run in sequence from top to bottom
	{
		ierr = MPI_Irecv(lxboundL_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Irecv(lxboundR_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		CurrPhTr->celly_xboundL[ny] = xboundL_pass[L][M][0];
		CurrPhTr->celly_xboundR[ny] = xboundR_pass[L][M][0];
	}

	//Send down from northernmost (top) to southmost (bottom)
	if(dsy->nproc != 1 && dsy->grprev != -1)
  	{
		for(lj = 0; lj < ny; lj++)
		{       
			xboundL_pass[L][M][lj] = CurrPhTr->celly_xboundL[lj];  
			xboundR_pass[L][M][lj] = CurrPhTr->celly_xboundR[lj];  
		}
		ierr = MPI_Isend(lxboundL_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Isend(lxboundR_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
  	}

	if(dsy->nproc != 1 && dsy->grprev != -1)  //Receive coordinates from previous node & set BOTTOM ghost node
	{
		ierr = MPI_Irecv(lxboundL_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Irecv(lxboundR_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grprev, 0, PETSC_COMM_WORLD, &rrequest); CHKERRQ(ierr);
		ierr = MPI_Wait(&rrequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
	
		CurrPhTr->celly_xboundL[-1] = xboundL_pass[L][M][ny-1];
		CurrPhTr->celly_xboundR[-1] = xboundR_pass[L][M][ny-1];
  	}

	if(dsy->nproc != 1 && dsy->grnext != -1)
  	{
		for(lj = 0; lj < ny; lj++)
		{       
			xboundL_pass[L][M][lj] = CurrPhTr->celly_xboundL[lj];  
			xboundR_pass[L][M][lj] = CurrPhTr->celly_xboundR[lj];  
		}
     	ierr = MPI_Isend(lxboundL_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     	ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);

		ierr = MPI_Isend(lxboundR_pass, (PetscMPIInt)(ny+1), MPIU_SCALAR, dsy->grnext, 0, PETSC_COMM_WORLD, &srequest); CHKERRQ(ierr);
     	ierr = MPI_Wait(&srequest, MPI_STATUSES_IGNORE);  CHKERRQ(ierr);
  	}

	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vxboundL_pass, &xboundL_pass); CHKERRQ(ierr);
	ierr = VecRestoreArray(vxboundL_pass, &lxboundL_pass); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vxboundL_pass); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(jr->DA_CELL_1D, vxboundR_pass, &xboundR_pass); CHKERRQ(ierr);
	ierr = VecRestoreArray(vxboundR_pass, &lxboundR_pass); CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(jr->DA_CELL_1D, &vxboundR_pass); CHKERRQ(ierr);

  PetscFunctionReturn(0);  
}
//----------------------------------------------------------------------------------------------------
// Set bottom bounds of NotInAir box based on solidus
// NOTE: this only tested with cpu_x = 1
//

PetscErrorCode Set_dike_base(JacRes *jr, PetscInt nD, PetscInt nPtr, PetscInt j1, PetscInt j2)
{

	FDSTAG      *fs;
	Dike        *dike;
	Discret1D   *dsz;
	FreeSurf    *surf;
	Ph_trans_t  *CurrPhTr;
	PetscScalar ***surface, ***solidus;
	PetscScalar dikeSolidus, xc, yc;
	PetscScalar localMinSolidus = PETSC_MAX_REAL;
	PetscScalar lithick, minLithoThick;
	PetscScalar localMinThickness = PETSC_MAX_REAL;
	PetscScalar yMinThick, loopx, xlt, localxlt, localyMinThickness;
	PetscScalar yMinSolidus, xms, localxms, localyMinSolidus;
	PetscInt    sx, nx, sy, ny, sz, nz, i, j, L;
	PetscInt    istep, nstep_out;

	PetscFunctionBeginUser;

	fs  =  jr->fs;
	dsz = &fs->dsz;
	L   =  (PetscInt)dsz->rank;

	istep=jr->ts->istep+1; 
	nstep_out=jr->ts->nstep_out;

	PetscCall(DMDAGetCorners(fs->DA_CEN, &sx, &sy, &sz, &nx, &ny, &nz));

	dike = jr->dbdike->matDike+nD;
	surf = jr->surf;
	CurrPhTr = jr->dbm->matPhtr+nPtr;

	// access work vectors
	PetscCall(DMDAVecGetArray(jr->DA_CELL_2D, dike->solidus, &solidus));
	PetscCall(DMDAVecGetArray(surf->DA_SURF, surf->gtopo, &surface));

	// find local minimum solidus and associated surface in dike zone nD
	for (j = j1; j <= j2; j++)
	{
		yc = COORD_CELL(j, sy, fs->dsy);
		localyMinThickness = PETSC_MAX_REAL;
		localyMinSolidus = PETSC_MAX_REAL;

		for (i = sx; i < sx + nx; i++)
		{
			xc = COORD_CELL(i, sx, fs->dsx);
			if (xc >= CurrPhTr->celly_xboundL[j] && xc <= CurrPhTr->celly_xboundR[j])
			{
				lithick = surface[L][j][i] - solidus[L][j][i]; // local index (lithospheric) thickness
				localMinThickness = PetscMin(localMinThickness, lithick);
				localMinSolidus = PetscMin(localMinSolidus, solidus[L][j][i]);

				// set local y-dependent minimum thickness
				if (lithick < localyMinThickness)
				{
					localyMinThickness = lithick; 
				}

				// set local y-dependent minimum solidus
				if (solidus[L][j][i] < localyMinSolidus)
				{
					localyMinSolidus = solidus[L][j][i];
				}
				
			}
		}

		// find y-location specific minimum lithospheric thickness and minimum solidus
			PetscCall(MPI_Allreduce(&localyMinThickness, &yMinThick, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD));
			PetscCall(MPI_Allreduce(&localyMinSolidus, &yMinSolidus, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD));
        
			// find associated coordinates
			localxlt = PETSC_MAX_REAL; // set really high so when reducing across procs we get the actual xlt-coord
			localxms = PETSC_MAX_REAL; // set really high so when reducing across procs we get the actual xms-coord

			for (i = sx; i < sx + nx; i++)
			{
				loopx = COORD_CELL(i, sx, fs->dsx);
				if (loopx >= CurrPhTr->celly_xboundL[j] && loopx <= CurrPhTr->celly_xboundR[j])
				{
					lithick = surface[L][j][i] - solidus[L][j][i]; 
					
					// local x-coords of minimum lithospheric thickness in y-location
					if (lithick == yMinThick)
					{
						localxlt = loopx;
					}

					// local x-coord minimum solidus (dike solidus by y-coord if this were not tied to entire dike)
					if (solidus[L][j][i] == yMinSolidus)
					{
						localxms = loopx;
					}
				}
			}

			// send coords to rank 0 processor
			PetscCall(MPI_Allreduce(&localxlt, &xlt, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD));
			PetscCall(MPI_Allreduce(&localxms, &xms, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD));

			// print info to file
			if (L == 0 && ((istep % nstep_out) == 0 || istep == 1))
			{
				std::ostringstream oss;
				oss << "dikeSolidus_Timestep_" << std::setfill('0') << std::setw(8) << (jr->ts->istep + 1) << ".txt";
				std::string filename = oss.str();

				// open output file
				::ofstream outFile(filename, std::ios_base::app); // append if already created
				if (outFile)
				{
					// write data [dike#, y-coord, minimum solidus (ms), xms, minimum lithospheric thickness (lt), xlt]
					outFile
						<< nD << " " << yc << " " << yMinSolidus << " " << xms
						<< " " << yMinThick << " " << xlt << "\n";
				}
				else
				{
					std::cerr << "Error opening file: " << filename << std::endl;
				}
		}
	}

	// find the minimum values across processors for this dike
	PetscCall(MPI_Allreduce(&localMinSolidus, &dikeSolidus, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD));
	PetscCall(MPI_Allreduce(&localMinThickness, &minLithoThick, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD));

	// restore access
	PetscCall(DMDAVecRestoreArray(jr->DA_CELL_2D, dike->solidus, &solidus));
	PetscCall(DMDAVecRestoreArray(surf->DA_SURF, surf->gtopo, &surface));

	// set zbounds[0] so that divergence only occurs in brittle lithosphere
	CurrPhTr->zbounds[0] = dikeSolidus;

	// solidus debug output *djking
	if (L==0 && ((istep % nstep_out) == 0 || istep == 1))
	{
		std::ostringstream oss;
		oss << "dikeSolidus_Timestep_" << std::setfill('0') << std::setw(8) << (jr->ts->istep + 1) << ".txt";
		std::string filename = oss.str();

		// open output file
		::ofstream outFile(filename, std::ios_base::app); // append if already created
		if (outFile)
		{
			// write data [dike#, dike solidus, minimum lithospheric thickness]
			outFile
				<< "dike " << nD << " dikeSolidus " << CurrPhTr->zbounds[0] 
				<< " minLithoThick " << minLithoThick << "\n";
		}
		else
		{
			std::cerr << "Error opening file: " << filename << std::endl;
		}
	}

  PetscFunctionReturn(0);  
}

//------------------------------------------------------------------------------------------------------------------
PetscErrorCode Compute_varDikingStress(JacRes *jr, PetscInt sFlag) // *djking
{

	Dike *dike;
	Ph_trans_t *CurrPhTr;
	FDSTAG *fs;
	PetscInt nD, numDike, numPhtr, nPtr, n;
	PetscInt j, j1, j2, sx, sy, sz, ny, nx, nz;
	PetscErrorCode ierr;

	PetscFunctionBeginUser;

	fs = jr->fs;

	PetscPrintf(PETSC_COMM_WORLD, "\n");
	numDike = jr->dbdike->numDike; // number of dikes
	numPhtr = jr->dbm->numPhtr;

	ierr = DMDAGetCorners(fs->DA_CEN, &sx, &sy, &sz, &nx, &ny, &nz); CHKERRQ(ierr);

	for (nD = 0; nD < numDike; nD++)
	{
		// access the parameters of the dike depending on the dike block
		dike = jr->dbdike->matDike + nD;

		// if there is any reason to find stress, magmatic pressure, or even the solidus
		if (dike->dyndike_start > 0 || jr->ctrl.var_M || jr->ctrl.sol_track)
		{
			//---------------------------------------------------------------------------------------------
			//  Find dike phase transition
			//---------------------------------------------------------------------------------------------
			nPtr = -1;
			for (n = 0; n < numPhtr; n++)
			{
				CurrPhTr = jr->dbm->matPhtr + n;
				if (CurrPhTr->ID == dike->PhaseTransID)
				{
					nPtr = n;
				}
			} // end loop over Phtr

			if (nPtr == -1)
				SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "PhaseTransID problems with dike %lld, nPtr=%lld\n", (LLD)(nD), (LLD)(nPtr));

			CurrPhTr = jr->dbm->matPhtr + nPtr;

			//---------------------------------------------------------------------------------------------
			//  Find y-bounds of current dynamic dike
			//---------------------------------------------------------------------------------------------
			j1 = ny - 1;
			j2 = 0;
			for (j = 0; j < ny; j++)
			{
				if (CurrPhTr->celly_xboundR[j] > CurrPhTr->celly_xboundL[j])
				{
					j1 = (PetscInt)min(j1, j);
					j2 = (PetscInt)max(j2, j);
				}
			}

			ierr = Compute_sxx_magP(jr, nD, sFlag); CHKERRQ(ierr); // compute mean effective sxx across the lithosphere

			ierr = Smooth_sxx_eff(jr, nD, nPtr, j1, j2, sFlag); CHKERRQ(ierr); // smooth mean effective sxx

		}
	}
	
	PetscFunctionReturn(0);
}

//---------------------------------------------------------------------------
PetscErrorCode DynamicDike_ReadRestart(DBPropDike *dbdike,  DBMat *dbm, JacRes *jr, TSSol *ts, FILE *fp)
{
	FB              *fb;
	Controls    *ctrl;
	Dike        *dike;
	PetscInt   nD, numDike;

	PetscErrorCode ierr;
	PetscFunctionBeginUser;

	ctrl = &jr->ctrl;
	if (!ctrl->actDike) PetscFunctionReturn(0);   // only execute this function if dikes are active

	ierr = FBLoad(&fb, PETSC_TRUE);
	ierr = TSSolCreate(ts, fb); 				CHKERRQ(ierr);

	numDike    = dbdike->numDike; // number of dikes

// create dike database
	ierr = DBDikeCreate(dbdike, dbm, fb, jr, PETSC_TRUE);   CHKERRQ(ierr);
	ierr = FBDestroy(&fb); CHKERRQ(ierr);

	for(nD = 0; nD < numDike; nD++)
	{
		dike = jr->dbdike->matDike+nD;

		if (dike->dyndike_start > 0 || jr->ctrl.var_M)
		{
			// read mean stress history, 2D array (local vector created with DA_CELL_2D_tave in DBReadDike)
			ierr = VecReadRestart(dike->sxx_eff_ave_hist, fp); CHKERRQ(ierr);
		}
	}

	PetscFunctionReturn(0);
}
//------------------------------------------------------------------------------------------------------------//
PetscErrorCode DynamicDike_WriteRestart(JacRes *jr, FILE *fp)
{

  Controls    *ctrl;
  Dike        *dike;
  PetscInt   nD, numDike;

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ctrl = &jr->ctrl;
  if (!ctrl->actDike) PetscFunctionReturn(0);   // only execute this function if dikes are active

  numDike    = jr->dbdike->numDike; // number of dikes

  for(nD = 0; nD < numDike; nD++)
  {
    dike = jr->dbdike->matDike+nD;
    if (dike->dyndike_start > 0 || jr->ctrl.var_M)
    {
      // WRITE mean stress history 2D array (local vector created with DA_CELL_2D_tave in DBReadDike)
       ierr = VecWriteRestart(dike->sxx_eff_ave_hist, fp); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}
  
//---------------------------------------------------------------------------

PetscErrorCode DynamicDike_Destroy(JacRes *jr)
{
  
  Dike        *dike;
  Controls    *ctrl;
  PetscInt   nD, numDike, dyndike_on;

  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ctrl = &jr->ctrl;

  if (!ctrl->actDike) PetscFunctionReturn(0);   // only execute this function if dikes are active


  numDike    = jr->dbdike->numDike; // number of dikes
  dyndike_on=0;
  for(nD = 0; nD < numDike; nD++)
  {
     dike = jr->dbdike->matDike+nD;
     if (dike->dyndike_start > 0 || jr->ctrl.var_M)
     {
       ierr = VecDestroy(&dike->sxx_eff_ave_hist); CHKERRQ(ierr);
       dyndike_on=1;
     }
  }

  if (dyndike_on==1)
  {
    ierr = DMDestroy(&jr->DA_CELL_2D_tave); CHKERRQ(ierr);
    ierr = DMDestroy(&jr->DA_CELL_1D); CHKERRQ(ierr);    
  }

  PetscFunctionReturn(0);
}

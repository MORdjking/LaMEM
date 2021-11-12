/*@ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 **
 **    Copyright (c) 2011-2015, JGU Mainz, Anton Popov, Boris Kaus
 **    All rights reserved.
 **
 **    This software was developed at:
 **
 **         Institute of Geosciences
 **         Johannes-Gutenberg University, Mainz
 **         Johann-Joachim-Becherweg 21
 **         55128 Mainz, Germany
 **
 **    project:    LaMEM                                                                                                                                                
 **    filename:   dike.cpp
 **
 **    LaMEM is free software: you can redistribute it and/or modify
 **    it under the terms of the GNU General Public License as published
 **    by the Free Software Foundation, version 3 of the License.
 **
 **    LaMEM is distributed in the hope that it will be useful,
 **    but WITHOUT ANY WARRANTY; without even the implied warranty of
 **    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 **    See the GNU General Public License for more details.
 **
 **    You should have received a copy of the GNU General Public License
 **    along with LaMEM. If not, see <http://www.gnu.org/licenses/>.
 **
 **
 **    Contact:
 **        Boris Kaus       [kaus@uni-mainz.de]
 **        Anton Popov      [popov@uni-mainz.de]
 **
 **
 **    This routine:
 **         Anton Popov      [popov@uni-mainz.de]
 **         Boris Kaus       [kaus@uni-mainz.de]
 **         Jana Schierjott
 **         Garrett Ito
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
//---------------------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DBDikeCreate"
PetscErrorCode DBDikeCreate(DBPropDike *dbdike, DBMat *dbm, FB *fb, PetscBool PrintOutput)   
{

        // read all dike parameter blocks from file
  
        PetscInt jj;

        PetscErrorCode ierr;
        PetscFunctionBegin;

        //===============                                                                                                                                               
        // DIKE PARAMETER                                                                                                               
        //===============                                                                                                                                               

        // setup block access mode                                                                                                                                      
        ierr = FBFindBlocks(fb, _OPTIONAL_, "<DikeStart>", "<DikeEnd>"); CHKERRQ(ierr);

        if(fb->nblocks)
        {
                // print overview of dike blocks from file                                                                                                           
            if (PrintOutput)
            {
		      PetscPrintf(PETSC_COMM_WORLD,"Dike blocks : \n");
            }
                // initialize ID for consistency checks                                                                                                                 

            for(jj = 0; jj < _max_num_dike_ ; jj++) dbdike->matDike[jj].ID = -1;

		// error checking
                if(fb->nblocks > _max_num_dike_)
                {
                        SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "Too many dikes specified! Max allowed: %lld", (LLD)_max_num_dike_);
                }

                // store actual number of dike blocks 
                dbdike->numDike = fb->nblocks;

                if (PrintOutput){
                        PetscPrintf(PETSC_COMM_WORLD,"--------------------------------------------------------------------------\n");
                }
		
                // read each individual dike block                                                                                                                   
                for(jj = 0; jj < fb->nblocks; jj++)
                {
                    ierr = DBReadDike(dbdike, dbm, fb, PrintOutput); CHKERRQ(ierr);
                    fb->blockID++;
                }
        }

	ierr = FBFreeBlocks(fb); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
//---------------------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DBReadDike"
PetscErrorCode DBReadDike(DBPropDike *dbdike, DBMat *dbm, FB *fb, PetscBool PrintOutput)
{
        // read dike parameter from file 
        Dike     *dike;
        PetscInt  ID;
	Scaling  *scal;
	
        PetscErrorCode ierr;
        PetscFunctionBegin;

	// access context
	scal    =  dbm->scal;
	
        // Dike ID                                                                                                                                                         
        ierr    = getIntParam(fb, _REQUIRED_, "ID", &ID, 1, dbdike->numDike-1); CHKERRQ(ierr);
        fb->ID  = ID;

        // get pointer to specified dike parameters
        dike = dbdike->matDike + ID;

        // check ID
        if(dike->ID != -1)
        {
                 SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Duplicate of Dike option!");
        }

        // set ID 
        dike->ID = ID;

	// read and store dike  parameters. 
        ierr = getScalarParam(fb, _REQUIRED_, "Mf",      &dike->Mf,      1, 1.0);              CHKERRQ(ierr);
        ierr = getScalarParam(fb, _REQUIRED_, "Mb",      &dike->Mb,      1, 1.0);              CHKERRQ(ierr);
	ierr = getIntParam(   fb, _REQUIRED_, "PhaseID", &dike->PhaseID, 1, dbm->numPhases-1); CHKERRQ(ierr);  
	ierr = getIntParam(   fb, _OPTIONAL_, "PhaseTransID", &dike->PhaseTransID, 1, dbm->numPhtr-1); CHKERRQ(ierr);
	ierr = getScalarParam(fb, _OPTIONAL_, "t0_dike", &dike->t0_dike, 1, 1.0);       CHKERRQ(ierr);
	ierr = getScalarParam(fb, _OPTIONAL_, "t1_dike", &dike->t1_dike, 1, 1.0);       CHKERRQ(ierr);
	ierr = getScalarParam(fb, _OPTIONAL_, "v_dike",  &dike->v_dike,  1, 1.0);   CHKERRQ(ierr);

	// scale parameters
      	dike->t0_dike /= scal->time;
       	dike->t1_dike /= scal->time;
	dike->v_dike  /= scal->velocity; 

        if (PrintOutput)
	    {
	    PetscPrintf(PETSC_COMM_WORLD,"   Dike parameters ID[%lld] : Mf = %g, Mb = %g\n", (LLD)(dike->ID), dike->Mf, dike->Mb);
      	    PetscPrintf(PETSC_COMM_WORLD,"   Optional dike parameters: v_dike = %g \n", dike->v_dike, scal->lbl_velocity);
	    PetscPrintf(PETSC_COMM_WORLD,"                             t0_dike = %g \n", dike->t0_dike, scal->lbl_time);
	    PetscPrintf(PETSC_COMM_WORLD,"                             t1_dike = %g \n", dike->t1_dike, scal->lbl_time);
	    PetscPrintf(PETSC_COMM_WORLD,"--------------------------------------------------------------------------\n");
        }

        PetscFunctionReturn(0);
}
//------------------------------------------------------------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "GetDikeContr"
PetscErrorCode GetDikeContr(ConstEqCtx *ctx,                                                                                                                                
                            PetscScalar *phRat,          // phase ratios in the control volume                                                                              
			    PetscScalar &dikeRHS,
                            PetscScalar &y_c)
{
  
  BCCtx       *bc;
  Dike        *dike;
  Ph_trans_t  *CurrPhTr;
  PetscInt     i, nD, nPtr, numDike, numPhtr;
  PetscScalar  v_spread, M, left, right, front, back;
  PetscScalar  y_distance, tempdikeRHS;
  
  numDike    = ctx->numDike;
  bc         = ctx->bc;
  numPhtr    = ctx->numPhtr;
  
  nPtr = 0;
  nD = 0;
  
  for(nPtr=0; nPtr<numPhtr; nPtr++)   // loop over all phase transitions blocks
    {
      // access the parameters of the phasetranstion block
      CurrPhTr = ctx->PhaseTrans+nPtr;
      
      for(nD = 0; nD < numDike; nD++) // loop through all dike blocks
	{
	  // access the parameters of the dike depending on the dike block
	  dike = ctx->matDike+nD;
	  
	  // access the phase ID of the dike parameters of each dike
	  i = dike->PhaseID;
	  
	  if(CurrPhTr->ID == dike->PhaseTransID)  // compare the phaseTransID associated with the dike with the actual ID of the phase transition in this cell           
	    {
	      
	      // check if the phase ratio of a dike phase is greater than 0 in the current cell
	      if(phRat[i]>0)
		{
		  PetscPrintf(PETSC_COMM_WORLD," PhaseTransID2 = %d \n", CurrPhTr->ID);
		  PetscPrintf(PETSC_COMM_WORLD," dikeID2 = %d \n", dike->PhaseTransID);
		  
		  if(dike->Mb == dike->Mf)  // constant M
		    {
		      M = dike->Mf;
		      v_spread = PetscAbs(bc->velin);
		      left = CurrPhTr->bounds[0];
		      right = CurrPhTr->bounds[1];
		      tempdikeRHS = M * 2 * v_spread / PetscAbs(left-right);
		    }
		  
		  else if(dike->Mb != dike->Mf)   // Mf and Mb are different
		    {
		      
		      PetscPrintf(PETSC_COMM_WORLD," y_c = %g \n", y_c);
		      
		      left = CurrPhTr->bounds[0];
		      right = CurrPhTr->bounds[1];
		      front = CurrPhTr->bounds[2];
		      back = CurrPhTr->bounds[3];
		      
		      v_spread = PetscAbs(bc->velin);
		      
		      // linear interpolation between different M values, Mf is M in front, Mb is M in back
		      y_distance = y_c - front;
		      PetscPrintf(PETSC_COMM_WORLD," front = %g \n", front);
		      PetscPrintf(PETSC_COMM_WORLD," back = %g \n", back);
		      PetscPrintf(PETSC_COMM_WORLD," left = %g \n", left);
		      PetscPrintf(PETSC_COMM_WORLD," right = %g \n", right);
		      PetscPrintf(PETSC_COMM_WORLD," y_distance = %g \n", y_distance);
		      PetscPrintf(PETSC_COMM_WORLD," v_spread = %g \n", v_spread);
		      
		      M = dike->Mf + (dike->Mb - dike->Mf) * (y_distance / (back - front));
		      PetscPrintf(PETSC_COMM_WORLD," M = %g \n", M);
		      
		      tempdikeRHS = M * 2 * v_spread / PetscAbs(left - right);
		      PetscPrintf(PETSC_COMM_WORLD," dikeRHS = %g \n", tempdikeRHS);
		      
		    }  // close loop Mb!=Mf
		  
		  else  // Mb and Mf don't exist (which should not occurr)
		    {
		      tempdikeRHS = 0.0;
		    }
		  
		  dikeRHS += phRat[i]*tempdikeRHS;
		  
		}  // close phase ratio loop
	    }  // close phase transition and phase ID comparison 
	}  // close dike block loop
    }  // close phase transition block loop
  
  PetscFunctionReturn(0);
}

//-----------------------------------------------------------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "Dike_k_heatsource"
PetscErrorCode Dike_k_heatsource(JacRes *jr,
                                 Material_t *phases,
                                 PetscScalar &Tc,
                                 PetscScalar *phRat,          // phase ratios in the control volume                                                                         
                                 PetscScalar &k,
                                 PetscScalar &rho_A)

{
  BCCtx       *bc;
  Dike        *dike;
  Ph_trans_t  *PhaseTrans;
  Material_t  *M;
  PetscInt     i, j, numDike;
  PetscScalar  v_spread, left, right, kfac, tempdikeRHS;
  
  numDike    = jr->dbdike->numDike; // number of dikes
  bc         =  jr->bc;
  PhaseTrans =  jr->dbm->matPhtr;   // phase transition
  
  j=0;
  
  // loop through all dikes
  for(j = 0; j < numDike; j++)
    {
      
      kfac = 0.0;
      
      //access the material parameters of each dike block
      dike=jr->dbdike->matDike+j;
      
      // access the phase ID of the dike block
      i = dike->PhaseID;
      
      // check if the phase ratio of a dike phase is greater than 0 in the current cell
      if(phRat[i] > 0)
	{
	  
	  if(dike->Mb == dike->Mf)
	    {
	      // constant M
	      v_spread = PetscAbs(bc->velin);
	      left = PhaseTrans->bounds[0];
	      right = PhaseTrans->bounds[1];
	      tempdikeRHS = dike->Mf * 2 * v_spread / PetscAbs(left-right);
	    }
	  
	  else
	    {
	      tempdikeRHS = 0.0;
	    } 
	  // end if (dike->Mb == dike-Mf)
	  
	  M = &phases[i];
	  
	  //adjust k and heat source according to Behn & Ito [2008]
	  if (Tc < M->T_liq && Tc > M->T_sol)
	    {
	      kfac  += phRat[i] / ( 1 + ( M->Latent_hx/ (M->Cp*(M->T_liq-M->T_sol))) );
	      rho_A += phRat[i]*(M->rho*M->Cp)*(M->T_liq-Tc)*tempdikeRHS;  // Cp*rho not used in the paper, added to conserve units of rho_A
	    }
	  else if (Tc <= M->T_sol)
	    {
	      kfac  += phRat[i];
	      rho_A += phRat[i]*( M->rho*M->Cp)*( (M->T_liq-Tc) + M->Latent_hx/M->Cp )*tempdikeRHS;
	    }
		else if (Tc >= M->T_liq)
		  {
		    kfac += phRat[i];
		  }
	  // end adjust k and heat source according to Behn & Ito [2008]
	  
	  k=kfac*k;
	  
	}   // end phase ratio
      
    }   // end dike loop
  
  PetscFunctionReturn(0);
}

//------------------------------------------------------------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "MovingDike"
PetscErrorCode MovingDike(DBPropDike *dbdike,
			  Ph_trans_t *PhaseTrans,
			  TSSol *ts)
{

  Dike        *dike;
  PetscInt     j, numDike;
  PetscScalar  t0_dike, t1_dike, v_dike;
  PetscScalar  t_current, dt;

  PetscFunctionBegin;//  NECESSARY?

  numDike    = dbdike->numDike;
  dt         = ts->dt;       // time step
  t_current  = ts->time;     // current time stamp, computed at the end of last time step round
  
  // loop through all dike blocks
  for(j = 0; j < numDike; j++)
    {
      
      // access the parameters of the dike depending on the dike block 
      dike = dbdike->matDike+j;
      
      // access the starting and end times of certain dike block
      t0_dike = dike->t0_dike;
      t1_dike = dike->t1_dike;
      v_dike  = dike->v_dike;

      // check if the current time step is equal to the starting time of when the dike is supposed to move
      if(t_current >= t0_dike && t_current <= t1_dike)
	{
	      
	  // condition for moving: phase transition ID needs to be the same as the Phase transitionID of the dike block
	  if(PhaseTrans->ID == dike->PhaseTransID)    
	    {
	      PhaseTrans->bounds[0] = PhaseTrans->bounds[0] + v_dike * dt;
	      PhaseTrans->bounds[1] = PhaseTrans->bounds[1] + v_dike * dt;
	    }
	  
	}
      
    }
  
  PetscFunctionReturn(0);

}
// --------------------------------------------------------------------------------------------------------------- 
